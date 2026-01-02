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
# Quantum LSTM 
# ----------------------------
class QuantumLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_i = QuantumLinear(input_dim + hidden_dim)  # Input Gate
        self.q_f = QuantumLinear(input_dim + hidden_dim)  # Forget Gate
        self.q_g = QuantumLinear(input_dim + hidden_dim)  # Candidate
        self.q_o = QuantumLinear(input_dim + hidden_dim)  # Output Gate

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)  # (B, input_dim + hidden_dim)

        i_t = self.sigmoid(self.q_i(combined)[:, :1])
        f_t = self.sigmoid(self.q_f(combined)[:, :1])
        g_t = self.tanh(self.q_g(combined)[:, :1])
        o_t = self.sigmoid(self.q_o(combined)[:, :1])

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

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
    def __init__(self, num_vars_k, num_vars_T, d_model=1, lstm_hidden=3):
        super(QuantumMyModel, self).__init__()

        self.vsn_k = QuantumVariableSelectionNetwork(num_vars=num_vars_k, d_model=d_model)
        self.vsn_T = QuantumVariableSelectionNetwork(num_vars=num_vars_T, d_model=d_model)

        self.lstm_cell = QuantumLSTMCell(input_dim=1, hidden_dim=lstm_hidden)
        self.lstm_cell = self.lstm_cell.to(torch.double)  # Ensure LSTM uses double precision

        self.glu1 = QuantumGLU(input_dim=1)
        self.grn1 = QuantumGRN(input_dim=1)
        self.attn = QuantumSingleHeadAttention(d_model=1)
        self.grn2 = QuantumGRN(input_dim=1)
        self.glu2 = QuantumGLU(input_dim=1)

        # Convert all quantum/classical modules to double
        self.vsn_k = self.vsn_k.to(torch.double)
        self.vsn_T = self.vsn_T.to(torch.double)
        self.glu1 = self.glu1.to(torch.double)
        self.grn1 = self.grn1.to(torch.double)
        self.attn = self.attn.to(torch.double)
        self.grn2 = self.grn2.to(torch.double)
        self.glu2 = self.glu2.to(torch.double)

    def forward(self, x_k, x_T):
        """
        x_k: Tensor of shape (B, k, num_vars_k)
        x_T: Tensor of shape (B, T, num_vars_T)
        """
        B, k, num_vars_k = x_k.shape
        _, T, num_vars_T = x_T.shape

        vsn_outputs = []

        # Step 1: VSN Encoding
        for t in range(k):
            xi = x_k[:, t, :].view(B, num_vars_k, 1)
            out, _ = self.vsn_k(xi)
            vsn_outputs.append(out)

        for t in range(T):
            xi = x_T[:, t, :].view(B, num_vars_T, 1)
            out, _ = self.vsn_T(xi)
            vsn_outputs.append(out)

        vsn_seq = torch.stack(vsn_outputs, dim=1)  # (B, k+T, 1)

        # Step 2: LSTM processing
        h_t = torch.zeros(B, 1, dtype=torch.double)
        c_t = torch.zeros(B, 1, dtype=torch.double)
        lstm_outs = []

        for t in range(k + T):
            x_t = vsn_seq[:, t, :]  # (B, 1)
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)  # (B, 1)
            lstm_outs.append(h_t)

        lstm_seq = torch.stack(lstm_outs, dim=1)  # (B, k+T, 1)

        # Step 3: Residual + GLU
        combined1 = self.glu1(lstm_seq) + vsn_seq  # (B, k+T, 1)

        # Step 4: GRN
        grn_out = self.grn1(combined1)  # (B, k+T, 1)

        # Step 5: Attention
        attn_out, _ = self.attn(grn_out)  # (B, k+T, 1)

        # Step 6: GLU + GRN again
        combined2 = self.glu1(attn_out) + grn_out
        grn_out2 = self.grn2(combined2)
        final = self.glu2(grn_out2) + combined1  # (B, k+T, 1)

        return final[:, -T:, :]  # (B, T, 1)
