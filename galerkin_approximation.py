from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Domain and mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 0.1), 100, 20)

# Function spaces
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Parameters
nu = 1.0e-6  # Viscosity of water at 20°C
sigma = 0.05  # Intensity of stochastic forcing

# Stochastic forcing term
W_noise = Expression(("sigma*sin(2*pi*x[0])*cos(2*pi*x[1])", 
                      "sigma*cos(2*pi*x[0])*sin(2*pi*x[1])"), 
                      degree=2, sigma=sigma)

# Boundary conditions
inlet_velocity = Constant((0.1, 0))
noslip = Constant((0, 0))
bc_inlet = DirichletBC(V, inlet_velocity, 'near(x[0], 0)')
bc_walls = DirichletBC(V, noslip, 'near(x[1], 0) || near(x[1], 0.1)')

# Variational form
F = (nu * inner(grad(u), grad(v)) * dx 
     - inner(p, div(v)) * dx 
     - inner(div(u), q) * dx
     + inner(W_noise, v) * dx)

a, L = lhs(F), rhs(F)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_inlet, bc_walls])

# Plot
plt.figure(figsize=(10, 2))
plot(u_sol, title="Velocity Field under Stochastic Forcing")
plt.show()
