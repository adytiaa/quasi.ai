import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, SIREN, GNPINN
from equations import NavierStokes2D
from utils import set_seed, count_parameters, save_model

# Set random seed for reproducibility
set_seed(42)

# Device configuration - use CPU for better compatibility
device = torch.device("cpu")
print(f"Using device: {device}")

# Define domain for crystal growth problem
domain_ranges = {
    'x': (0, 1),     # Horizontal dimension of the domain
    'y': (0, 1),     # Vertical dimension of the domain
    't': (0, 0.5),   # Time domain
}

# Define a custom crystal interface function that evolves with time
# This represents a growing crystal with a slightly curved interface
def crystal_interface_fn(x, t):
    # Base position
    base_height = 0.2
    
    # Small curvature in the interface (hump in the middle)
    curvature = 0.05 * torch.sin(np.pi * x)
    
    # Growth rate - crystal grows upward with time
    growth_rate = 0.1 * t
    
    return base_height + curvature + growth_rate

# Create the Navier-Stokes equation for crystal growth
navier_stokes_eq = NavierStokes2D(
    domain_ranges=domain_ranges,
    viscosity=0.01,                    # Fluid viscosity
    thermal_diffusivity=0.005,         # Thermal diffusivity
    density=1.0,                       # Fluid density
    crystal_interface_fn=crystal_interface_fn,
    device=device
)

# Neural network parameters
input_dim = 3      # (x, y, t)
output_dim = 4     # (u, v, p, T) - velocities, pressure, temperature
hidden_layers = 4  # Reduced for faster training
neurons_per_layer = 50  # Reduced for faster training

# Create neural network - SIREN works well for fluid dynamics
model = SIREN(input_dim, output_dim, hidden_layers, neurons_per_layer)
print(f"Model has {count_parameters(model)} trainable parameters")

# Create GNPINN
pinn = GNPINN(model, navier_stokes_eq, device)
print("Using Gradient-Normalized PINN for crystal growth simulation")

# Optimizer
learning_rate = 1e-3  # Increased for faster training
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training parameters - reduced for testing
n_epochs = 50  # Reduced for faster testing
n_collocation_points = 2000  # Reduced for faster testing
log_interval = 10

# Create output directory
os.makedirs("results", exist_ok=True)

# Training
print(f"Training model on {device}...")
history = pinn.train(
    optimizer,
    n_collocation_points=n_collocation_points,
    n_epochs=n_epochs,
    log_interval=log_interval
)
print("Training complete!")

# Save model
save_path = os.path.join("results", "crystal_growth_model.pt")
metadata = {
    'equation': 'navier_stokes_crystal',
    'domain_ranges': domain_ranges,
    'viscosity': navier_stokes_eq.viscosity,
    'thermal_diffusivity': navier_stokes_eq.thermal_diffusivity,
    'hidden_layers': hidden_layers,
    'neurons_per_layer': neurons_per_layer,
    'final_losses': {k: v[-1] for k, v in history.items()}
}
save_model(model, save_path, metadata)
print(f"Model saved to {save_path}")

# Plot loss history
try:
    plt.figure(figsize=(10, 6))
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join("results", "crystal_growth_loss.png"))
    print("Created loss plot.")
except Exception as e:
    print(f"Could not create loss plot: {e}")

# Function to visualize flow field and temperature at a specific time
def visualize_crystal_growth(model, equation, time_value, resolution=50, plot_streamlines=True):
    """
    Visualize the flow field and temperature distribution for crystal growth.
    
    Args:
        model: Trained neural network model
        equation: NavierStokes2D equation object
        time_value: Time at which to visualize
        resolution: Grid resolution for visualization
        plot_streamlines: Whether to plot velocity streamlines
    """
    try:
        # Check if numpy is available and working
        try:
            import numpy as np
            numpy_available = True
        except (ImportError, ValueError, RuntimeError):
            numpy_available = False
            print("NumPy not available, using PyTorch for visualization")
        
        # Extract domain boundaries
        x_min, x_max = equation.domain_ranges['x']
        y_min, y_max = equation.domain_ranges['y']
        
        # Create spatial grid
        x = torch.linspace(x_min, x_max, resolution, device=device)
        y = torch.linspace(y_min, y_max, resolution, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Get crystal interface at this time
        interface_y = equation.crystal_interface(X.flatten(), time_value).reshape(X.shape)
        
        # Create input points
        X_flat = X.flatten().unsqueeze(1)
        Y_flat = Y.flatten().unsqueeze(1)
        T_flat = torch.ones_like(X_flat) * time_value
        
        points = torch.cat([X_flat, Y_flat, T_flat], dim=1)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            outputs = model(points)
            u = outputs[:, 0].reshape(X.shape)
            v = outputs[:, 1].reshape(X.shape)
            p = outputs[:, 2].reshape(X.shape)
            T = outputs[:, 3].reshape(X.shape)
        
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        if numpy_available:
            # Convert to numpy for plotting
            try:
                X_np = X.cpu().numpy()
                Y_np = Y.cpu().numpy()
                u_np = u.cpu().numpy()
                v_np = v.cpu().numpy()
                p_np = p.cpu().numpy()
                T_np = T.cpu().numpy()
                interface_y_np = interface_y.cpu().numpy()
            
                # Create a mask for points below the crystal interface
                mask = Y_np < interface_y_np
                
                # Apply mask to velocity field and temperature
                u_masked = np.ma.array(u_np, mask=mask)
                v_masked = np.ma.array(v_np, mask=mask)
                T_masked = np.ma.array(T_np, mask=mask)
                
                # Calculate velocity magnitude safely to avoid negative values
                vel_magnitude = np.sqrt(np.maximum(0, u_masked**2 + v_masked**2))
                
                # Plot 1: Velocity field
                cm1 = axs[0].pcolormesh(X_np, Y_np, vel_magnitude, 
                                      cmap='viridis', shading='auto')
                plt.colorbar(cm1, ax=axs[0], label='Velocity magnitude')
                
                # Add streamlines if requested
                if plot_streamlines:
                    try:
                        # Subsample grid for streamlines
                        stride = 2
                        # For streamplot, we need the original coordinates which are guaranteed to be strictly increasing
                        x_stream = x.cpu().numpy()[::stride]  # Subsample from original linspace
                        y_stream = y.cpu().numpy()[::stride]  # Subsample from original linspace
                        
                        # Prepare the velocity field with matching dimensions
                        u_stream = u_masked[::stride, ::stride].T
                        v_stream = v_masked[::stride, ::stride].T
                        
                        axs[0].streamplot(x_stream, y_stream, u_stream, v_stream,
                                       color='white', density=1.0, linewidth=0.5, arrowsize=0.5)
                    except Exception as e:
                        print(f"Streamplot error: {e}")
                        # Fall back to quiver plot
                        axs[0].quiver(X_np[::3, ::3], Y_np[::3, ::3], 
                                    u_np[::3, ::3], v_np[::3, ::3], 
                                    color='white', scale=30)
                
                # Plot crystal interface
                axs[0].plot(X_np[0, :], interface_y_np[0, :], 'k-', linewidth=2)
                
                # Fill the crystal region
                axs[0].fill_between(X_np[0, :], interface_y_np[0, :], Y_np.min(), color='gray', alpha=0.5)
                
                # Plot 2: Pressure field
                cm2 = axs[1].pcolormesh(X_np, Y_np, p_np, cmap='RdBu_r', shading='auto')
                plt.colorbar(cm2, ax=axs[1], label='Pressure')
                
                # Plot crystal interface
                axs[1].plot(X_np[0, :], interface_y_np[0, :], 'k-', linewidth=2)
                axs[1].fill_between(X_np[0, :], interface_y_np[0, :], Y_np.min(), color='gray', alpha=0.5)
                
                # Plot 3: Temperature field
                cm3 = axs[2].pcolormesh(X_np, Y_np, T_masked, cmap='hot', shading='auto')
                plt.colorbar(cm3, ax=axs[2], label='Temperature')
                
                # Plot crystal interface
                axs[2].plot(X_np[0, :], interface_y_np[0, :], 'k-', linewidth=2)
                axs[2].fill_between(X_np[0, :], interface_y_np[0, :], Y_np.min(), color='gray', alpha=0.5)
            
            except Exception as e:
                print(f"NumPy-based plotting failed: {e}")
                numpy_available = False
        
        # Fallback to PyTorch-based plotting if NumPy fails
        if not numpy_available:
            # Use PyTorch tensors directly
            X_cpu = X.cpu()
            Y_cpu = Y.cpu()
            u_cpu = u.cpu()
            v_cpu = v.cpu()
            p_cpu = p.cpu()
            T_cpu = T.cpu()
            interface_y_cpu = interface_y.cpu()
            
            # Create a mask for the crystal region (as boolean tensor)
            mask = Y_cpu < interface_y_cpu
            
            # Create a simple velocity magnitude tensor, masking crystal region
            vel_magnitude = torch.sqrt(torch.clamp(u_cpu**2 + v_cpu**2, min=0.0))
            vel_magnitude = vel_magnitude.masked_fill(mask, 0)
            
            # Plot using tensor values converted to lists
            # Plot 1: Velocity field (simplified)
            scatter1 = axs[0].scatter(X_cpu.flatten().tolist(), Y_cpu.flatten().tolist(), 
                                   c=vel_magnitude.flatten().tolist(), cmap='viridis', s=1)
            plt.colorbar(scatter1, ax=axs[0], label='Velocity magnitude')
            
            # Simple quiver plot with subsampling instead of streamlines
            if plot_streamlines:
                stride = 4  # More aggressive subsampling for quiver
                axs[0].quiver(X_cpu[::stride, ::stride].flatten().tolist(),
                           Y_cpu[::stride, ::stride].flatten().tolist(),
                           u_cpu[::stride, ::stride].flatten().tolist(),
                           v_cpu[::stride, ::stride].flatten().tolist(),
                           color='white', scale=30, width=0.001)
            
            # Plot crystal interface
            axs[0].plot(X_cpu[0, :].tolist(), interface_y_cpu[0, :].tolist(), 'k-', linewidth=2)
            
            # Plot 2: Pressure (simplified)
            scatter2 = axs[1].scatter(X_cpu.flatten().tolist(), Y_cpu.flatten().tolist(), 
                                   c=p_cpu.flatten().tolist(), cmap='RdBu_r', s=1)
            plt.colorbar(scatter2, ax=axs[1], label='Pressure')
            axs[1].plot(X_cpu[0, :].tolist(), interface_y_cpu[0, :].tolist(), 'k-', linewidth=2)
            
            # Plot 3: Temperature (simplified)
            T_masked = T_cpu.masked_fill(mask, 0)
            scatter3 = axs[2].scatter(X_cpu.flatten().tolist(), Y_cpu.flatten().tolist(), 
                                   c=T_masked.flatten().tolist(), cmap='hot', s=1)
            plt.colorbar(scatter3, ax=axs[2], label='Temperature')
            axs[2].plot(X_cpu[0, :].tolist(), interface_y_cpu[0, :].tolist(), 'k-', linewidth=2)
        
        # Set labels and titles
        for i, title in enumerate(['Velocity Field', 'Pressure Field', 'Temperature Field']):
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            axs[i].set_title(f'{title} at t={time_value:.2f}')
        
        plt.tight_layout()
        return fig, axs
    except Exception as e:
        print(f"Visualization error: {e}")
        # Create a simple figure as fallback
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f"Visualization failed: {e}", 
                horizontalalignment='center', verticalalignment='center')
        return fig, ax

# Visualize results at different time steps
try:
    time_points = [0.0, 0.2, 0.4]  # Reduced number of time points for testing
    
    for t in time_points:
        fig, _ = visualize_crystal_growth(model, navier_stokes_eq, t, resolution=30)  # Lower resolution for faster plotting
        plt.savefig(os.path.join("results", f"crystal_growth_t{t:.1f}.png"))
        plt.close(fig)
    
    print("Created visualization plots")
except Exception as e:
    print(f"Could not create visualization: {e}")

print("Crystal growth simulation complete!") 