import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class PINN:
    """
    Physics-Informed Neural Network base class
    """
    
    def __init__(self, model, pde, device=None):
        """
        Initialize PINN.
        
        Args:
            model: Neural network model
            pde: PDE object that defines the equation, boundary conditions, etc.
            device: Torch device (cpu or cuda)
        """
        self.model = model
        self.pde = pde
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
    def pde_loss(self, x):
        """
        Calculate PDE residual loss.
        
        Args:
            x: Collocation points
            
        Returns:
            Mean squared residual
        """
        residual = self.pde.compute_residual(self.model, x)
        return torch.mean(residual**2)
    
    def bc_ic_loss(self):
        """
        Calculate boundary and initial condition loss.
        
        Returns:
            Combined boundary and initial condition MSE loss
        """
        bc = self.pde.get_boundary_conditions()
        ic = self.pde.get_initial_conditions()
        
        bc_points, bc_values = bc['points'], bc['values']
        ic_points, ic_values = ic['points'], ic['values']
        
        # Predict values at boundary and initial points
        bc_pred = self.model(bc_points)
        ic_pred = self.model(ic_points)
        
        # Calculate mean squared error
        bc_mse = torch.mean((bc_pred - bc_values)**2)
        ic_mse = torch.mean((ic_pred - ic_values)**2)
        
        return bc_mse + ic_mse
    
    def total_loss(self, x):
        """
        Calculate total loss.
        
        Args:
            x: Collocation points
            
        Returns:
            Dictionary containing PDE loss, BC/IC loss, and total loss
        """
        pde_loss = self.pde_loss(x)
        bc_ic_loss = self.bc_ic_loss()
        total = pde_loss + bc_ic_loss
        
        return {
            'pde_loss': pde_loss,
            'bc_ic_loss': bc_ic_loss,
            'total_loss': total
        }
    
    def train(self, optimizer, n_collocation_points=10000, n_epochs=10000, log_interval=100):
        """
        Train the PINN model.
        
        Args:
            optimizer: PyTorch optimizer
            n_collocation_points: Number of collocation points for PDE residual
            n_epochs: Number of training epochs
            log_interval: How often to log progress
            
        Returns:
            Dictionary of training history
        """
        history = {'pde_loss': [], 'bc_ic_loss': [], 'total_loss': []}
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Generate collocation points
            x = self.pde.generate_collocation_points(n_collocation_points)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute losses
            losses = self.total_loss(x)
            
            # Backward pass and optimizer step
            losses['total_loss'].backward()
            optimizer.step()
            
            # Record losses
            history['pde_loss'].append(losses['pde_loss'].item())
            history['bc_ic_loss'].append(losses['bc_ic_loss'].item())
            history['total_loss'].append(losses['total_loss'].item())
            
            # Log progress
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{n_epochs}, "
                      f"PDE Loss: {losses['pde_loss'].item():.4e}, "
                      f"BC/IC Loss: {losses['bc_ic_loss'].item():.4e}, "
                      f"Total Loss: {losses['total_loss'].item():.4e}, "
                      f"Time: {elapsed_time:.2f}s")
                
        return history


class GNPINN(PINN):
    """
    Gradient-Normalized Physics-Informed Neural Network
    """
    
    def train(self, optimizer, n_collocation_points=10000, n_epochs=10000, log_interval=100):
        """
        Train the GNPINN model using gradient normalization.
        
        Args:
            optimizer: PyTorch optimizer
            n_collocation_points: Number of collocation points for PDE residual
            n_epochs: Number of training epochs
            log_interval: How often to log progress
            
        Returns:
            Dictionary of training history
        """
        history = {'pde_loss': [], 'bc_ic_loss': [], 'total_loss': []}
        
        for epoch in range(n_epochs):
            start_time = time.time()
            self.model.train()
            
            # Generate collocation points
            x = self.pde.generate_collocation_points(n_collocation_points)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate PDE loss
            pde_loss = self.pde_loss(x)
            
            # Calculate BC/IC loss
            bc_ic_loss = self.bc_ic_loss()
            
            # Total loss (not used for backpropagation in GN-PINN)
            total_loss = pde_loss + bc_ic_loss
            
            try:
                # --- Gradient Normalization ---
                # Get gradients for PDE loss
                pde_grads = torch.autograd.grad(pde_loss, self.model.parameters(), 
                                              retain_graph=True, create_graph=True, allow_unused=True)
                
                # Get gradients for BC/IC loss
                bc_ic_grads = torch.autograd.grad(bc_ic_loss, self.model.parameters(), 
                                                retain_graph=True, create_graph=True, allow_unused=True)
                
                # Normalize gradients
                normalized_pde_grads = []
                for g in pde_grads:
                    if g is not None:
                        norm = torch.norm(g)
                        if norm > 1e-8:  # Avoid division by very small numbers
                            normalized_pde_grads.append(g / norm)
                        else:
                            normalized_pde_grads.append(g)
                    else:
                        normalized_pde_grads.append(None)
                
                normalized_bc_ic_grads = []
                for g in bc_ic_grads:
                    if g is not None:
                        norm = torch.norm(g)
                        if norm > 1e-8:  # Avoid division by very small numbers
                            normalized_bc_ic_grads.append(g / norm)
                        else:
                            normalized_bc_ic_grads.append(g)
                    else:
                        normalized_bc_ic_grads.append(None)
                
                # Combine normalized gradients
                final_grads = []
                for i in range(len(pde_grads)):
                    if pde_grads[i] is not None and bc_ic_grads[i] is not None:
                        final_grads.append(normalized_pde_grads[i] + normalized_bc_ic_grads[i])
                    elif pde_grads[i] is not None:
                        final_grads.append(normalized_pde_grads[i])
                    elif bc_ic_grads[i] is not None:
                        final_grads.append(normalized_bc_ic_grads[i])
                    else:
                        final_grads.append(None)
                
                # Apply the final gradients back to the parameters
                for param, grad in zip(self.model.parameters(), final_grads):
                    if grad is not None:
                        param.grad = grad
            
            except RuntimeError as e:
                # Fallback to standard backprop if gradient normalization fails
                print(f"Warning: Gradient normalization failed, using standard backprop. Error: {e}")
                optimizer.zero_grad()
                total_loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record losses
            history['pde_loss'].append(pde_loss.item())
            history['bc_ic_loss'].append(bc_ic_loss.item())
            history['total_loss'].append(total_loss.item())
            
            # Log progress
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{n_epochs}, "
                      f"PDE Loss: {pde_loss.item():.4e}, "
                      f"BC/IC Loss: {bc_ic_loss.item():.4e}, "
                      f"Total Loss: {total_loss.item():.4e}, "
                      f"Time: {elapsed_time:.2f}s")
                
        return history 