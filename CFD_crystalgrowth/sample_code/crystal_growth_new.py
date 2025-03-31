import numpy as np
import matplotlib.pyplot as plt

def simulate_crystal_growth(time_steps=1000, grid_size=100):
    """
    Simulate silicon single crystal growth using a simple phase-field model.
    
    Parameters:
        time_steps (int): Number of simulation steps.
        grid_size (int): Size of the simulation grid.
    
    Returns:
        growth_field (ndarray): Final state of the crystal growth field.
    """
    
    # Initialize a 2D grid with a seed crystal at the center
    growth_field = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    growth_field[center - 1:center + 1, center - 1:center + 1] = 1  # Seed crystal

    # Define parameters for growth dynamics
    growth_rate = 0.1
    diffusion_rate = 0.01

    for t in range(time_steps):
        # Compute diffusion and growth dynamics
        laplacian = (
            np.roll(growth_field, 1, axis=0) +
            np.roll(growth_field, -1, axis=0) +
            np.roll(growth_field, 1, axis=1) +
            np.roll(growth_field, -1, axis=1) -
            4 * growth_field
        )
        
        growth_field += diffusion_rate * laplacian + growth_rate * (1 - growth_field)
        
        # Apply boundary conditions (zero flux)
        growth_field[0, :] = growth_field[-1, :] = growth_field[:, 0] = growth_field[:, -1] = 0

        if t % (time_steps // 10) == 0:
            print(f"Time step {t}/{time_steps}")

    return growth_field

# Run the simulation and visualize results
if __name__ == "__main__":
    result = simulate_crystal_growth()
    
    plt.imshow(result, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Silicon Single Crystal Growth")
    plt.show()

