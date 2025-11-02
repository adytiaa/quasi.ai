import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

from models import MLP, SIREN, PINN, GNPINN
from equations import Heat1D, KdV, NavierStokes2D
from utils import set_seed, count_parameters, save_model, load_model
from visualization import plot_loss_history, plot_solution_1d


def run_heat_equation(args):
    """
    Run Heat equation experiment
    """
    print("\n=== Heat Equation Experiment ===")
    
    # Define domain
    domain_ranges = {
        'x': (0, 1),  # Spatial domain: x ∈ [0, 1]
        't': (0, 1),  # Time domain: t ∈ [0, 1]
    }
    
    # Create PDE instance
    heat_eq = Heat1D(domain_ranges=domain_ranges, diffusivity=args.diffusivity, device=args.device)
    
    # Create neural network
    if args.network_type == 'mlp':
        model = MLP(2, 1, args.hidden_layers, args.neurons, activation=args.activation)
    elif args.network_type == 'siren':
        model = SIREN(2, 1, args.hidden_layers, args.neurons)
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")
    
    print(f"Model has {count_parameters(model)} trainable parameters")
    
    # Create PINN
    if args.use_gn:
        pinn = GNPINN(model, heat_eq, args.device)
        print("Using Gradient-Normalized PINN")
    else:
        pinn = PINN(model, heat_eq, args.device)
        print("Using standard PINN")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training
    print(f"Training model on {args.device}...")
    history = pinn.train(
        optimizer,
        n_collocation_points=args.collocation_points,
        n_epochs=args.epochs,
        log_interval=args.log_interval
    )
    print("Training complete!")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, "heat_model.pt")
        metadata = {
            'equation': 'heat',
            'args': vars(args),
            'final_losses': {k: v[-1] for k, v in history.items()}
        }
        save_model(model, save_path, metadata)
        print(f"Model saved to {save_path}")
    
    # Plot results
    try:
        # Plot loss history
        plot_loss_history(history)
        plt.savefig(os.path.join(args.output_dir, "heat_loss.png"))
        
        # Plot solution
        print("Plotting solution...")
        plot_solution_1d(model, heat_eq)
        plt.savefig(os.path.join(args.output_dir, "heat_solution.png"))
        
        if args.show_plots:
            plt.show()
    except Exception as e:
        print(f"Warning: Error during plotting: {e}")
        print("Plotting failed, but model training was successful.")
        
        # Try simpler plot for loss history
        try:
            plt.figure(figsize=(10, 6))
            plt.semilogy(history['total_loss'], label='Total Loss')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, "heat_loss_simple.png"))
            print("Created simplified loss plot.")
        except Exception as e2:
            print(f"Could not create simplified plot: {e2}")

def run_thermal_copling(args):
    """
    Run thermal-fluid coupling simulation for silicon melt
    """
    print("\n=== Thermal-Fluid Coupling Simulation ===")
    
    # Define domain
    domain_ranges = {
        'x': (-5, 5),   # Spatial domain: x ∈ [-5, 5]
        'y': (-5, 5),   # Spatial domain: y ∈ [-5, 5]
        't': (0, 1),    # Time domain: t ∈ [0, 1]
    }
    
    # Create PDE instance
    thermal_fluid_eq = NavierStokesEnergy(domain_ranges=domain_ranges, device=args.device)
    
    # Create neural network
    if args.network_type == 'mlp':
        model = MLP(3, 4, args.hidden_layers, args.neurons, activation=args.activation)  # 3 inputs (x, y, t), 4 outputs (u, v, p, T)
    elif args.network_type == 'siren':
        model = SIREN(3, 4, args.hidden_layers, args.neurons)
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")
    
    print(f"Model has {count_parameters(model)} trainable parameters")
    
    # Create PINN
    if args.use_gn:
        pinn = GNPINN(model, thermal_fluid_eq, args.device)
        print("Using Gradient-Normalized PINN")
    elif args.use_rf:
        pinn = rf-pinn(model, thermal_fluid_eq, args.device)
        print("Using reactive flow PINN")
    else:
        pinn = PINN(model, thermal_fluid_eq, args.device)
        print("Using standard PINN")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training
    print(f"Training model on {args.device}...")
    history = pinn.train(
        optimizer,
        n_collocation_points=args.collocation_points,
        n_epochs=args.epochs,
        log_interval=args.log_interval
    )
    print("Training complete!")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, "thermal_fluid_model.pt")
        metadata = {
            'equation': 'navier-stokes-energy',
            'args': vars(args),
            'final_losses': {k: v[-1] for k, v in history.items()}
        }
        save_model(model, save_path, metadata)
        print(f"Model saved to {save_path}")
    
    # Plot results
    try:
        # Plot loss history
        plot_loss_history(history)
        plt.savefig(os.path.join(args.output_dir, "thermal_fluid_loss.png"))
        
        # Plot velocity and temperature fields
        def plot_fluid_fields(n_points=100, n_times=5):
            x_min, x_max = domain_ranges['x']
            y_min, y_max = domain_ranges['y']
            t_min, t_max = domain_ranges['t']
            
            x = torch.linspace(x_min, x_max, n_points, device=args.device)
            y = torch.linspace(y_min, y_max, n_points, device=args.device)
            times = torch.linspace(t_min, t_max, n_times, device=args.device)
            
            X, Y = torch.meshgrid(x, y, indexing='xy')
            points = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            plt.figure(figsize=(12, 10))
            
            for i, t_value in enumerate(times):
                t_tensor = torch.full((points.shape[0], 1), t_value, device=args.device)
                input_points = torch.cat([points, t_tensor], dim=1)
                
                model.eval()
                with torch.no_grad():
                    output = model(input_points)
                    u, v, T = output[:, 0], output[:, 1], output[:, 3]
                
                plt.subplot(n_times, 1, i+1)
                plt.quiver(X.cpu(), Y.cpu(), u.cpu().reshape(n_points, n_points), v.cpu().reshape(n_points, n_points))
                plt.title(f'Velocity Field at t = {t_value.item():.2f}')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid(True)
            
            plt.tight_layout()
        
        print("Plotting solution...")
        plot_fluid_fields()
        plt.savefig(os.path.join(args.output_dir, "thermal_fluid_evolution.png"))
        
        if args.show_plots:
            plt.show()
    except Exception as e:
        print(f"Warning: Error during plotting: {e}")
        print("Plotting failed, but model training was successful.")



def run_kdv_equation(args):
    """
    Run KdV equation experiment
    """
    print("\n=== KdV Equation Experiment ===")
    
    # Define domain
    domain_ranges = {
        'x': (-10, 10),  # Spatial domain: x ∈ [-10, 10]
        't': (0, 5),     # Time domain: t ∈ [0, 5]
    }
    
    # Create PDE instance
    kdv_eq = KdV(domain_ranges=domain_ranges, beta=args.beta, device=args.device)
    
    # Create neural network
    if args.network_type == 'mlp':
        model = MLP(2, 1, args.hidden_layers, args.neurons, activation=args.activation)
    elif args.network_type == 'siren':
        model = SIREN(2, 1, args.hidden_layers, args.neurons)
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")
    
    print(f"Model has {count_parameters(model)} trainable parameters")
    
    # Create PINN
    if args.use_gn:
        pinn = GNPINN(model, kdv_eq, args.device)
        print("Using Gradient-Normalized PINN")
    else:
        pinn = PINN(model, kdv_eq, args.device)
        print("Using standard PINN")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training
    print(f"Training model on {args.device}...")
    history = pinn.train(
        optimizer,
        n_collocation_points=args.collocation_points,
        n_epochs=args.epochs,
        log_interval=args.log_interval
    )
    print("Training complete!")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, "kdv_model.pt")
        metadata = {
            'equation': 'kdv',
            'args': vars(args),
            'final_losses': {k: v[-1] for k, v in history.items()}
        }
        save_model(model, save_path, metadata)
        print(f"Model saved to {save_path}")
    
    # Plot results
    try:
        # Plot loss history
        plot_loss_history(history)
        plt.savefig(os.path.join(args.output_dir, "kdv_loss.png"))
        
        # Plot solution evolution
        def plot_kdv_evolution(n_points=200, n_times=5):
            x_min, x_max = kdv_eq.domain_ranges['x']
            t_min, t_max = kdv_eq.domain_ranges['t']
            
            # Create spatial grid
            x = torch.linspace(x_min, x_max, n_points, device=args.device)
            
            # Create time points
            times = torch.linspace(t_min, t_max, n_times, device=args.device)
            
            plt.figure(figsize=(12, 10))
            
            for i, t_value in enumerate(times):
                # Create time tensor
                t_tensor = torch.ones_like(x, device=args.device) * t_value
                
                # Create input points
                points = torch.stack([x, t_tensor], dim=1)
                
                # Evaluate model
                model.eval()
                with torch.no_grad():
                    try:
                        u_pred = model(points).cpu().numpy().flatten()
                        u_exact = kdv_eq.exact_solution(x, t_value).cpu().numpy()
                        x_np = x.cpu().numpy()
                    except RuntimeError:
                        u_pred = model(points).cpu().flatten()
                        u_exact = kdv_eq.exact_solution(x, t_value).cpu()
                        x_np = x.cpu()
                
                # Plot solution
                plt.subplot(n_times, 1, i+1)
                try:
                    plt.plot(x_np, u_pred, 'r-', label='PINN')
                    plt.plot(x_np, u_exact, 'b--', label='Exact')
                except TypeError:
                    plt.plot(x_np.tolist(), u_pred.tolist(), 'r-', label='PINN')
                    plt.plot(x_np.tolist(), u_exact.tolist(), 'b--', label='Exact')
                plt.title(f'KdV Solution at t = {t_value.item():.2f}')
                plt.xlabel('x')
                plt.ylabel('u(x,t)')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
        
        print("Plotting solution...")
        plot_kdv_evolution()
        plt.savefig(os.path.join(args.output_dir, "kdv_evolution.png"))
        
        if args.show_plots:
            plt.show()
    except Exception as e:
        print(f"Warning: Error during plotting: {e}")
        print("Plotting failed, but model training was successful.")
        
        # Try simpler plot for loss history
        try:
            plt.figure(figsize=(10, 6))
            plt.semilogy(history['total_loss'], label='Total Loss')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, "kdv_loss_simple.png"))
            print("Created simplified loss plot.")
        except Exception as e2:
            print(f"Could not create simplified plot: {e2}")


def run_crystal_growth(args):
    """
    Run the crystal growth simulation using Navier-Stokes equations
    """
    # Set the device
    device = torch.device(args.device)

    # Define domain for crystal growth problem
    domain = {
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
        curvature = 0.05 * torch.sin(torch.pi * x)
        
        # Growth rate - crystal grows upward with time
        growth_rate = 0.1 * t
        
        return base_height + curvature + growth_rate

    # Create the Navier-Stokes equation for crystal growth
    ns_eq = NavierStokes2D(
        domain_ranges=domain,
        viscosity=args.viscosity,
        thermal_diffusivity=args.thermal_diffusivity,
        density=args.density,
        crystal_interface_fn=crystal_interface_fn,
        device=device
    )

    # Neural network parameters
    input_dim = 3      # (x, y, t)
    output_dim = 4     # (u, v, p, T) - velocities, pressure, temperature

    # Create neural network - SIREN works well for fluid dynamics
    if args.network_type == 'mlp':
        model = MLP(
            input_dim,
            output_dim,
            args.hidden_layers,
            args.neurons,
            activation=args.activation
        )
    else:  # SIREN
        model = SIREN(
            input_dim,
            output_dim,
            args.hidden_layers,
            args.neurons
        )

    # Print model parameters
    print(f"Model has {count_parameters(model)} trainable parameters")

    # Create the PINN
    if args.use_gn:
        pinn = GNPINN(model, ns_eq, device)
        print("Using Gradient-Normalized PINN for crystal growth simulation")
    else:
        pinn = PINN(model, ns_eq, device)
        print("Using standard PINN for crystal growth simulation")

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    print(f"Training model on {device}...")
    history = pinn.train(
        optimizer,
        n_collocation_points=args.collocation_points,
        n_epochs=args.epochs,
        log_interval=args.log_interval
    )
    print("Training complete!")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the model
    save_path = os.path.join(args.output_dir, "crystal_growth_model.pt")
    metadata = {
        'equation': 'navier_stokes_crystal',
        'domain_ranges': domain,
        'viscosity': ns_eq.viscosity,
        'thermal_diffusivity': ns_eq.thermal_diffusivity,
        'density': ns_eq.density,
        'network_type': args.network_type,
        'hidden_layers': args.hidden_layers,
        'neurons': args.neurons,
        'use_gn': args.use_gn,
        'final_losses': {k: v[-1] for k, v in history.items()}
    }
    save_model(model, save_path, metadata)
    print(f"Model saved to {save_path}")

    # Plot loss history
    try:
        plt.figure(figsize=(10, 6))
        plt.semilogy(history['total_loss'], label='Total Loss')
        plt.semilogy(history['pde_loss'], label='PDE Loss')
        plt.semilogy(history['bc_ic_loss'], label='BC/IC Loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PINN Training Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "crystal_growth_loss.png"))
        print("Created loss plot.")
    except Exception as e:
        print(f"Could not create loss plot: {e}")

    # Function to visualize flow field and temperature at a specific time
    def visualize_crystal_growth(model, equation, time_value, resolution=50, plot_streamlines=True):
        """
        Visualize the flow field and temperature distribution for crystal growth.
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
        time_points = [0.0, 0.2, 0.4]
        
        for t in time_points:
            fig, _ = visualize_crystal_growth(model, ns_eq, t, resolution=args.plot_resolution)
            plt.savefig(os.path.join(args.output_dir, f"crystal_growth_t{t:.1f}.png"))
            plt.close(fig)
        
        print("Created visualization plots")
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    """
    Main function to parse arguments and run experiments
    """
    parser = argparse.ArgumentParser(description="PINN Framework for PDEs")
    
    # General arguments
    parser.add_argument('--equation', type=str, default='heat', choices=['heat', 'schrodinger', 'kdv', 'crystal'],
                        help='PDE to solve')
    parser.add_argument('--use_gn', action='store_true', help='Use Gradient-Normalized PINN')
    parser.add_argument('--network_type', type=str, default='mlp', choices=['mlp', 'siren'],
                        help='Neural network architecture')
    parser.add_argument('--hidden_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--neurons', type=int, default=50,
                        help='Neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'gelu', 'sine'],
                        help='Activation function (for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--collocation_points', type=int, default=10000,
                        help='Number of collocation points for PDE residual')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda, or cuda:0)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available (overrides --device)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots')
    
    # Equation-specific arguments
    parser.add_argument('--diffusivity', type=float, default=1.0,
                        help='Diffusivity coefficient for Heat equation')
    parser.add_argument('--beta', type=float, default=0.0025,
                        help='Beta coefficient for KdV equation')
    
    # Navier-Stokes specific parameters
    parser.add_argument('--viscosity', type=float, default=0.01, help='Kinematic viscosity (for Navier-Stokes)')
    parser.add_argument('--thermal_diffusivity', type=float, default=0.005, 
                        help='Thermal diffusivity (for Navier-Stokes)')
    parser.add_argument('--density', type=float, default=1.0, help='Fluid density (for Navier-Stokes)')
    parser.add_argument('--plot_resolution', type=int, default=40, help='Resolution for plots')
    
    args = parser.parse_args()
    
    # Set device - default to CPU unless --use_gpu is specified
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    print(f"Using device: {args.device}")
    
    # Set random seed
    set_seed(args.seed)

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        print("Warning: matplotlib is not available. Plotting will be disabled.")
        has_matplotlib = False
        args.show_plots = False

    # Check if numpy is available
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        print("Warning: numpy is not available. Some features may be limited.")
        has_numpy = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected experiment
    if args.equation == 'heat':
        run_heat_equation(args)
    elif args.equation == 'thermal_copling':
        run_thermal_copling(args)
    elif args.equation == 'kdv':
        run_kdv_equation(args)
    elif args.equation == 'crystal':
        run_crystal_growth(args)
    elif args.equation == 'crystal_new':
        run_crystal_growth_new(args)    
    else:
        raise ValueError(f"Unknown equation: {args.equation}")


if __name__ == "__main__":
    main() 
