# PINN Variants for Enhancing Accuracy and Efficiency of CFD for Silicon Single Crystal Growth Simulations

A modular framework implementing variants of Physics-Informed Neural Networks (PINNs) for thermal-fluid coupling which includes Navier-Stokes equations CFD for crystal growth simulations.

## Features

- **Modular Design**: Easily extendable framework for PINNs with clean separation of concerns
- **Gradient Normalization**: Improved training stability using advanced gradient normalization techniques
- **Multiple PDE Support**:
  - Navier-Stokes Equations for Crystal Growth (2D)
  - Navier-Stokes Equations for Crystal Growth (3D)
- **Neural Network Options**:
  - MLP (Multi-Layer Perceptron)
  - SIREN (Sinusoidal Representation Networks)
- **Visualization Tools**: Comprehensive plotting utilities for solution visualization

The thermal-fluid coupling phenomenon in silicon melt is governed by the Navier-Stokes equations, which describe the conservation of mass, momentum, and energy in fluid flow. These equations are coupled with the energy equation to account for heat transfer and the species transport equation for solute diffusion in the case of binary mixtures like silicon-germanium melts[1][3][4].

**Governing Equations:**

1. **Conservation of Mass (Continuity Equation):**
   $$
   \nabla \cdot \mathbf{u} = 0
   $$
   This equation ensures that the fluid is incompressible, meaning the density remains constant.

2. **Conservation of Momentum (Navier-Stokes Equation):**
   $$
   \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F}
   $$
   Here, $$\mathbf{u}$$ is the velocity vector, $$p$$ is the pressure, $$\mu$$ is the dynamic viscosity, and $$\mathbf{F}$$ represents external forces, such as gravitational or electromagnetic forces.

3. **Conservation of Energy:**
   $$
   \rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T) + Q
   $$
   $$T$$ is the temperature, $$c_p$$ is the specific heat capacity, $$k$$ is the thermal conductivity, and $$Q$$ represents heat sources or sinks.

4. **Species Transport Equation (for binary mixtures):**
   $$
   \frac{\partial C}{\partial t} + \mathbf{u} \cdot \nabla C = D \nabla^2 C
   $$
   $$C$$ is the concentration of the solute (e.g., silicon in germanium melt), and $$D$$ is the diffusion coefficient.

**Relation to Navier-Stokes Equations:**
The Navier-Stokes equations form the core of the thermal-fluid coupling model, describing the fluid flow and its interaction with thermal and species transport phenomena. The energy and species transport equations are coupled with the Navier-Stokes equations through the velocity field $$\mathbf{u}$$, which influences both the temperature distribution and the solute concentration in the melt[1][3][4]. This coupling is essential for accurately modeling processes like crystal growth, where thermal gradients and fluid flow significantly impact the quality of the resulting material.

Citations:
[1] https://dergipark.org.tr/tr/download/article-file/1181337
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC11243553/
[3] https://joam.inoe.ro/articles/numerical-modeling-of-thermo-physical-properties-on-molten-si-in-multi-crystalline-silicon-growth-process/fulltext
[4] https://www.comsol.it/paper/download/46209/Mechighel.pdf
[5] https://www.degruyter.com/document/doi/10.1515/zna-2023-0293/pdf
[6] https://www.scirp.org/journal/paperinformation?paperid=43127
[7] https://www.tandfonline.com/doi/full/10.1080/10910344.2024.2400956?src=exp-la
[8] https://www.mdpi.com/1996-1944/17/24/6287

---
Answer from Perplexity: https://www.perplexity.ai/search/hermal-fluid-coupling-phenomen-VPXgpjZER.W8DFHmczF3lA?utm_source=copy_output
