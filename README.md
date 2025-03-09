# quasi  

Quantum Machine Learning for Enhancing Fluid Dynamic Simulations using PINNs and Quantum PINNs

## Idea
Leveraging quantum machine learning techniques to enhance fluid dynamic simulations using Physics-Informed Neural Networks (PINNs) and their quantum counterparts (Quantum PINNs or QPINNs). The study will explore various implementations, including attention-enhanced architectures and Physically-Informed Quantum Neural Networks (PI-QNNs). The aim is to demonstrate the potential of quantum computing in solving high-dimensional partial differential equations (PDEs) associated with fluid dynamics, improving computational efficiency and accuracy.

Open source frameworks pennylane (QML), nvidia Modulus (PINNs)
### https://pennylane.ai/  
### https://developer.nvidia.com/modulus 

### Objectives
1. Develop and evaluate classical PINNs for fluid dynamics simulations.
2. Design and implement Quantum PINNs (QPINNs) leveraging quantum circuits and hybrid quantum-classical models.
3. Explore attention-enhanced PINNs to improve feature extraction and representation.
4. Investigate Physically-Informed Quantum Neural Networks (PI-QNNs) to directly incorporate physical constraints into quantum models.
5. Benchmark the performance of these models in terms of accuracy, efficiency, and scalability.

### Methodology

#### 1. Classical PINNs
- Implement baseline PINNs using standard deep learning frameworks such as TensorFlow and PyTorch.
- Solve benchmark fluid dynamics problems, including the Navier-Stokes equations for incompressible flow.
- Evaluate accuracy and computational performance on varying problem scales.

#### 2. Quantum PINNs (QPINNs)
- Utilize hybrid quantum-classical platforms like TensorFlow Quantum or Qiskit to design QPINNs.
- Employ PQCs to encode problem features and leverage variational quantum algorithms for optimization.
- Train models using quantum simulators and available quantum hardware (e.g., IBM Quantum).

#### 3. Attention-Enhanced PINNs
- Integrate multi-head attention mechanisms to focus on critical regions of the solution domain.
- Test attention layers for improving convergence rates and accuracy in learning complex dynamics.

#### 4. Physically-Informed Quantum Neural Networks (PI-QNNs)
- Develop quantum neural networks with loss functions explicitly encoding PDE constraints.
- Explore the use of entanglement to represent coupled physical systems.
- Validate models on multi-scale and high-dimensional fluid dynamics problems.

#### 5. Benchmarking Framework
- Evaluate classical, attention-enhanced, and quantum models on metrics such as:
  - **Accuracy**: Compare predicted solutions against analytical or high-fidelity numerical solutions.
  - **Efficiency**: Assess computational cost on quantum and classical hardware.
  - **Scalability**: Examine performance for increasing dimensionality and complexity.


#### Quantum Circuits for Quantum PINNs
At the heart of QPINNs are parameterized quantum circuits (PQCs), which offer a way to model complex, high-dimensional functions with fewer parameters compared to classical neural networks. These circuits enable the encoding of PDE solutions as quantum states, with optimization performed using hybrid quantum-classical approaches. 

- **Architecture:** QPINNs employ hardware-efficient ansatz designs tailored to specific fluid dynamics problems, incorporating gates that respect the problem’s symmetries.
- **Attention Mechanisms:** Attention layers are integrated within PQCs to focus computational resources on critical fluid features, such as vortices or shockwaves, enhancing solution accuracy.
- **Optimization:** Loss functions based on the residuals of governing PDEs are minimized using hybrid gradient methods, such as the Parameter Shift Rule or Quantum Natural Gradient Descent.

#### Tensor Networks and Matrix Product States
Tensor networks, including Matrix Product States (MPS), play a crucial role in encoding the high-dimensional states of fluid dynamics simulations. MPS are particularly efficient for problems with localized interactions, allowing scalable representation of complex flow fields.

1. **Decomposition:** The computational domain is divided into overlapping subdomains, each represented as a tensor, enabling localized modeling of fluid behavior.
2. **Quantum Embedding:** Tensor network states are mapped onto quantum circuits to exploit quantum entanglement for enhanced representational power.
3. **Integration with QPINNs:** MPS-based data compression is combined with PQCs, ensuring efficient handling of large simulation domains.

#### QMERA for Multi-Scale Fluid Dynamics
Multi-scale Entanglement Renormalization Ansatz (QMERA) extends tensor network techniques to hierarchical modeling of fluid systems. QMERA is particularly suited for multi-scale phenomena, such as turbulence, where both fine-grained and coarse-grained features must be captured efficiently.

- **Hierarchical Encoding:** QMERA constructs a renormalized representation of fluid states, iteratively coarse-graining fine details while preserving global features.
- **Renormalization Layers:** These layers reduce the effective dimensionality of the problem, enabling efficient quantum optimization.
- **Applications:** Turbulent flows and multi-phase interactions have been effectively simulated using QMERA-enhanced QPINNs, demonstrating superior accuracy compared to classical PINNs.

#### Attention-Enhanced Mechanisms
Attention mechanisms in QPINNs prioritize critical regions of the solution space, such as high-gradient areas in fluid flow. This is achieved through quantum-inspired attention layers, which:
- Dynamically reallocate computational resources during training.
- Enhance convergence rates by focusing on regions where PDE residuals are highest.
- Integrate seamlessly with tensor networks and quantum circuits for hybrid optimization.



Some reference papers which we are looking at in detail 
#### https://www.mdpi.com/1099-4300/26/8/649 
#### https://arxiv.org/html/2304.11247v3    
#### http://arxiv.org/pdf/2406.18749.pdf    
#### https://pubmed.ncbi.nlm.nih.gov/39202119/        


## Contact

Name: A Seshaditya 

E-Mail: aditya@zedat.fu-berlin.de 

Website: https://adytiaa.github.io/quasi.ai/ 

LinkedIn: https://www.linkedin.com/in/a-seshaditya-7180822a5/  


## License

This project is licensed under the MIT License. See the LICENSE file for details.

