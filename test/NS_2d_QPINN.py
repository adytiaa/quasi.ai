# Solving PDEs using Quantum Physics-Inspired Neural Network approaches

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Device and quantum circuit settings
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Ansatz: basic hardware-efficient circuit
def quantum_circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (3, n_qubits, 3)}

# Hybrid model class for 2D Navier-Stokes
class QuantumPINN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.linear_in = nn.Linear(2, n_qubits)  # 2D input: x, y

    def forward(self, x):
        x_in = self.linear_in(x)
        return self.qlayer(x_in)

# Residuals for 2D Navier-Stokes (incompressible, steady-state)
def residual_2d(model, xy):
    xy = xy.clone().detach().requires_grad_(True)
    psi = model(xy)

    grads = torch.autograd.grad(psi, xy, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    u = grads[:, 1]  # du/dy
    v = -grads[:, 0]  # -du/dx

    u_x = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]
    v_x = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1]

    # Continuity equation (divergence-free)
    continuity = u_x + v_y
    
    # Assume viscosity = 1 for simplicity
    residual_u = u * u_x + v * u_y - (torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0] +
                                      torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1])

    residual_v = u * v_x + v * v_y - (torch.autograd.grad(v_x, xy, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0] +
                                      torch.autograd.grad(v_y, xy, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1])

    return continuity, residual_u, residual_v

# Training setup for 2D
model_2d = QuantumPINN2D()
optimizer = torch.optim.Adam(model_2d.parameters(), lr=0.01)

xy_train = torch.rand((100, 2)) * 2 - 1  # random 2D domain between -1 and 1
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    cont, res_u, res_v = residual_2d(model_2d, xy_train)
    loss = torch.mean(cont**2) + torch.mean(res_u**2) + torch.mean(res_v**2)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# 3D Navier-Stokes model
class QuantumPINN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.linear_in = nn.Linear(3, n_qubits)  # 3D input: x, y, z

    def forward(self, x):
        x_in = self.linear_in(x)
        return self.qlayer(x_in)

# Residuals for 3D Navier-Stokes
def residual_3d(model, xyz):
    xyz = xyz.clone().detach().requires_grad_(True)
    psi = model(xyz)

    grads = torch.autograd.grad(psi, xyz, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    u = grads[:, 1]
    v = -grads[:, 0]
    w = grads[:, 2]

    u_x = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
    u_y = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]
    u_z = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2]

    v_x = torch.autograd.grad(v, xyz, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0]
    v_y = torch.autograd.grad(v, xyz, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1]
    v_z = torch.autograd.grad(v, xyz, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 2]

    w_x = torch.autograd.grad(w, xyz, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 0]
    w_y = torch.autograd.grad(w, xyz, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 1]
    w_z = torch.autograd.grad(w, xyz, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 2]

    continuity = u_x + v_y + w_z

    residual_u = u * u_x + v * u_y + w * u_z - (torch.autograd.grad(u_x, xyz, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0] +
                                                torch.autograd.grad(u_y, xyz, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1] +
                                                torch.autograd.grad(u_z, xyz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 2])

    residual_v = u * v_x + v * v_y + w * v_z - (torch.autograd.grad(v_x, xyz, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0] +
                                                torch.autograd.grad(v_y, xyz, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1] +
                                                torch.autograd.grad(v_z, xyz, grad_outputs=torch.ones_like(v_z), create_graph=True)[0][:, 2])

    residual_w = u * w_x + v * w_y + w * w_z - (torch.autograd.grad(w_x, xyz, grad_outputs=torch.ones_like(w_x), create_graph=True)[0][:, 0] +
                                                torch.autograd.grad(w_y, xyz, grad_outputs=torch.ones_like(w_y), create_graph=True)[0][:, 1] +
                                                torch.autograd.grad(w_z, xyz, grad_outputs=torch.ones_like(w_z), create_graph=True)[0][:, 2])

    return continuity, residual_u, residual_v, residual_w

# Visualization of 2D velocity field
xy_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20), indexing='ij'), dim=-1).view(-1, 2)
with torch.no_grad():
    psi_grid = model_2d(xy_grid)
    grads = torch.autograd.grad(psi_grid, xy_grid, grad_outputs=torch.ones_like(psi_grid), create_graph=False)[0]
    u = grads[:, 1].view(20, 20)
    v = -grads[:, 0].view(20, 20)

plt.figure(figsize=(6, 5))
plt.quiver(xy_grid[:, 0].view(20, 20), xy_grid[:, 1].view(20, 20), u, v)
plt.title("2D Velocity Field")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

