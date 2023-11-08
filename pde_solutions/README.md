Code snippets and notebooks for producing reference (numerical and analytical) solutions to the PDE we are interested in. 

## Convection (linear advection)

$$
  \frac{\partial u}{\partial t} + \beta \frac{\partial u}{\partial x} = 0
  \qquad
  x \in [0,M], \, t \in [0,T]
$$

With initial condition: 

$$
  u(x,0) = h(x)
  \qquad
  x \in [0,M]
$$

and periodic boundary condition: 

$$
  u(0,t) = u(M,t)
  \qquad
  t \in [0,T]
$$

The notebook `convection.ipynb` includes the referenced [MAT file](https://github.com/AdityaLab/pinnsformer/blob/main/demo/convection/convection.mat) from [PINNsFormer paper](https://arxiv.org/abs/2307.11833), Fourier Transform [method](https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py) from the [PINNs failure mode paper](https://arxiv.org/abs/2109.01050), Finite Difference (Lax-Wendroff) method, and an analytical solution.  
