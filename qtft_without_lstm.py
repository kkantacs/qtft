import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# ----------------------------
# Quantum Linear Layer
# ----------------------------
class QuantumLinear(nn.Module):
    def __init__(self, input_dim, n_layers=2):
        super().__init__()
        self.input_dim = input_dim

        dev = qml.device("default.qubit", wires=input_dim)

        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(input_dim))
            qml.templates.BasicEntanglerLayers(weights, wires=range(input_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(input_dim)]

        qnode = qml.QNode(circuit, dev, interface="torch")
        weight_shapes = {"weights": (n_layers, input_dim)}
        self.layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.layer(x)

# ----------------------------
# Quantum GLU
# ----------------------------
class QuantumGLU(nn.Module):
    def __init__(self, input_dim, n_layers=2):
        super().__init__()
        self.q_gate = QuantumLinear(input_dim, n_layers)
        self.q_feature = QuantumLinear(input_dim, n_layers)

    def forward(self, x):
        x = x.to(torch.double).requires_grad_()
        gate = torch.sigmoid(self.q_gate(x))
        feature = self.q_feature(x)
        return gate * feature

# ----------------------------
# Quantum GRN
# ----------------------------
class QuantumGRN(nn.Module):
    def __init__(self, input_dim, n_layers=2):
        super().__init__()
        self.q_linear_a = QuantumLinear(input_dim, n_layers)
        self.q_linear_eta2 = QuantumLinear(input_dim, n_layers)
        self.qglu = QuantumGLU(input_dim, n_layers)

    def forward(self, a, c=None):
        a = a.to(torch.double).requires_grad_()
        eta1 = F.elu(self.q_linear_a(a))
        eta2 = self.q_linear_eta2(eta1)
        eta3 = self.qglu(eta2)
        return a + eta3

# ----------------------------
# Quantum Variable Selection Network
# ----------------------------
class QuantumVariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, d_model):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.selection_grn = QuantumGRN(num_vars * d_model)
        self.variable_grn = QuantumGRN(d_model)

    def forward(self, xi_t, context=None):
        B, m_x, d_model = xi_t.shape
        xi_flat = xi_t.view(B, -1)
        logits = self.selection_grn(xi_flat, context)
        v_weights = F.softmax(logits.view(B, m_x), dim=-1)

        xi_proc = self.variable_grn(xi_t.view(B * m_x, d_model)).view(B, m_x, d_model)
        output = torch.sum(v_weights.unsqueeze(-1) * xi_proc, dim=1)
        return output, v_weights

# ----------------------------
# Quantum Single Head Attention
# ----------------------------
class QuantumSingleHeadAttention(nn.Module):
    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.q_query = QuantumLinear(d_model, n_layers)
        self.q_key = QuantumLinear(d_model, n_layers)
        self.q_value = QuantumLinear(d_model, n_layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D).to(torch.double)
        Q = self.q_query(x_flat).view(B, T, D)
        K = self.q_key(x_flat).view(B, T, D)
        V = self.q_value(x_flat).view(B, T, D)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output, attn_weights

# ----------------------------
# Full Quantum TFT Model
# ----------------------------
class QuantumMyModel(nn.Module):
    def __init__(self, num_vars_k, num_vars_T, d_model=1, lstm_hidden=1):
        super().__init__()
        self.vsn_k = QuantumVariableSelectionNetwork(num_vars_k, d_model)
        self.vsn_T = QuantumVariableSelectionNetwork(num_vars_T, d_model)
        self.lstm = nn.LSTMCell(1, lstm_hidden).to(torch.double)
        self.glu1 = QuantumGLU(1)
        self.grn1 = QuantumGRN(1)
        self.attn = QuantumSingleHeadAttention(1)
        self.grn2 = QuantumGRN(1)
        self.glu2 = QuantumGLU(1)

    def forward(self, x_k, x_T):
        B, k, _ = x_k.shape
        T = x_T.shape[1]
        vsn_outs = []

        for t in range(k):
            out, _ = self.vsn_k(x_k[:, t:t+1, :].view(B, -1, 1))
            vsn_outs.append(out)
        for t in range(T):
            out, _ = self.vsn_T(x_T[:, t:t+1, :].view(B, -1, 1))
            vsn_outs.append(out)

        vsn_seq = torch.stack(vsn_outs, dim=1)
        h_t = torch.zeros(B, 1, dtype=vsn_seq.dtype)
        c_t = torch.zeros(B, 1, dtype=vsn_seq.dtype)

        lstm_outs = []
        for t in range(k + T):
            h_t, c_t = self.lstm(vsn_seq[:, t, :], (h_t, c_t))
            lstm_outs.append(h_t)

        lstm_seq = torch.stack(lstm_outs, dim=1)
        combined1 = self.glu1(lstm_seq) + vsn_seq
        grn_out = self.grn1(combined1)
        attn_out, _ = self.attn(grn_out)
        combined2 = self.glu1(attn_out) + grn_out
        grn_out2 = self.grn2(combined2)
        final = self.glu2(grn_out2) + combined1
        return final[:, -T:, :]
