# quasi  

Quantum Machine Learning for Enhancing Fluid Dynamic Simulations 

## Idea
Leveraging quantum machine learning techniques to enhance fluid dynamic simulations using 



### Objectives


### Methodology

#### 1. Classical PINNs
- Implement baseline PINNs using standard deep learning frameworks such as TensorFlow and PyTorch.
- Solve benchmark fluid dynamics problems, including the Navier-Stokes equations for incompressible flow.
- Evaluate accuracy and computational performance on varying problem scales.

#### 2. Quantum PINNs (QPINNs)
- Utilize hybrid quantum-classical platforms like TensorFlow Quantum or Qiskit or PennyLane to design QPINNs.
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
At the core are parameterized quantum circuits (PQCs), which offer a way to model complex, high-dimensional functions with fewer parameters compared to classical neural networks. These circuits enable the encoding of PDE solutions as quantum states, with optimization performed using hybrid quantum-classical approaches. 

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

#### Frameworks and Resouces
   Pennylane, Nvidia PhysicsNeMo, DGX Quantum, Qiskit, IBM Quantum
https://pennylane.ai/ https://developer.nvidia.com/physicsnemo https://quantum.ibm.com/

#### References
Some reference papers are in the Quantum and Physics-Informed AI models

Afrah Farea, Saiful Khan, & Mustafa Serdar Celebi. (2025). QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs. arXiv:2503.16678 [quant-ph]. https://arxiv.org/abs/2503.16678


Hybrid Quantum PINNs for Computational Fluid Dynamics. (2025). IOP Science. https://doi.org/10.1088/2632-2153/ad43b2


Quantum Physics-Informed Machine Learning for Multiscale Simulations. (2024). arXiv:2403.08954 [quant-ph]. https://arxiv.org/abs/2403.08954


PINNSFORMER: A Transformer-Based Framework for Physics-Informed Neural Networks. (2023). arXiv:2307.11833v3 [cs.LG].


Pengpeng Xiao, Muqing Zheng, Anran Jiao, Xiu Yang, & Lu Lu. (2024). Quantum DeepONet: Neural Operators Accelerated by Quantum Computing. [Manuscript].


Siddhant Dutta, Nouhaila Innan, Sadok Ben Yahia, & Muhammad Shafique. (2024). AQ-PINNs: Attention-Enhanced Quantum Physics-Informed Neural Networks for Carbon-Efficient Climate Modeling. arXiv:2409.01626v1 [cs.LG].


Physics-Informed Neural Networks in Crystal Growth Simulations
D. Kolberg, C. Anders, & S. Reke. (2023). Fast Prediction of Transport Structures in the Melt by Physics-Informed Neural Networks during ‘VMCz’ Crystal Growth of Silicon. Journal of the Institute of Metals, https://doi.org/10.1080/00219592.2023.2236656


J. Ling, Y. Zhang, & D. Chen. (2023). Simulation of Thermal‑Fluid Coupling in Silicon Single Crystal Growth Based on Gradient Normalized Physics‑Informed Neural Network. Physics of Fluids. https://doi.org/10.1063/5.0123811
 (Code repository: https://github.com/callmedrcom/SIPINN)


Z. Wang, F. Chen, & M. Li. (2024). RF-PINNs: Reactive Flow Physics-Informed Neural Networks for Field Reconstruction of Laminar and Turbulent Flames. Journal of Computational Physics, 113698. https://doi.org/10.1016/j.jcp.2024.113698


Y. Zhao, H. Liu, & J. Xu. (2025). Research on the Thermal‑Fluid Coupling in the Growth Process of Czochralski Silicon Single Crystals Based on an Improved Physics‑Informed Neural Network. AIP Advances, 15(10), 105202. https://doi.org/10.1063/5.0271778



## Contact

Name: A Seshaditya 

E-Mail: aditya.a.sesh@alumni.tu-berlin.de, aditya@quasi.digital 

Website: https://adytiaa.github.io/quasi.ai/  
         https://quasi.digital

LinkedIn: https://www.linkedin.com/in/a-seshaditya-7180822a5/  


## License

This project is licensed under the MIT License. See the LICENSE file for details.
