import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for reactive flow.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PINN, self).__init__()
        
        # Define the neural network architecture
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def physics_loss(u, v, T, p, C, x, y, t):
    """
    Define the governing equations as residuals for the loss function.
    u: velocity in x-direction
    v: velocity in y-direction
    T: temperature
    p: pressure
    C: concentration
    x, y: spatial coordinates
    t: time
    """
    # Compute gradients using PyTorch's autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    
    # Continuity equation (incompressibility)
    continuity_residual = u_x + v_y

    # Momentum equations (Navier-Stokes)
    momentum_residual_u = u_t + u * u_x + v * u_y + p + 1.0 * (u_x + v_y)

    # Energy equation
    energy_residual = T_t + u * T + 1.0

    return continuity_residual.mean()**2 + momentum_residual_u.mean()**2 + energy_residual.mean()**2


