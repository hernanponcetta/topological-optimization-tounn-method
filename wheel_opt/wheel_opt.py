import time
import os

from fenics import *
import numpy as np
import torch
import torch.optim as optim
from ufl import relabel

from msh2xdmf.msh2xdmf import import_mesh

from nn.top_optimizer_nn import TopOptimizerNN
from nn.top_opt_loss import TopOptLoss
from utils.utils import create_mid_points, create_time_stamp, weight_init, write_optimization_data, write_to_csv, write_optimization_data

NAME = "wheel_2d"

# Start time for optimization
start = time.time()

# Set FEniCS log level to ERROR
set_log_level(40)

#Loging interval
log_interval = 1

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create time stamp
ts = create_time_stamp()

# Create output directory
os.makedirs("output/{name}/{ts}/".format(ts=ts, name=NAME))

# Directory name for data
data_directory = "output/{name}/{ts}".format(ts=ts, name=NAME)

# Create file for saving the results
xdmf = XDMFFile("{data_directory}/{name}_density.xdmf".format(data_directory=data_directory,
                                                              name=NAME))

# Parameters
# Relation between original volume and target volume
volume_ratio = 0.1

# Penalty parameter for density
penal = 3

# Parameters for the neuron network and optminizer
learning_rate = 0.001
nrm_threshold = 0.1
min_epochs = 20
max_epochs = 3000

dim = 2

# Parameter for the penalty term, more info:
# https://en.wikipedia.org/wiki/Penalty_method
# https://www.youtube.com/watch?v=RTEpONXUJyE&ab_channel=ChristopherLum
alpha_max = 100 * volume_ratio
alpha_increment = 0.05
alpha = alpha_increment

# Import mesh, more info:
# https://github.com/floiseau/msh2xdmf
mesh, boundaries_mf, association_table = import_mesh(
    prefix='mesh',
    dim=dim,
    directory="./mesh/",
)

# Prepare finite element analisis
mu = Constant(0.3)
lmbda = Constant(0.6)
sigma = lambda u: 2.0 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(len(u))
psi = lambda u: lmbda / 2 * (tr(sym(grad(u)))**2) + mu * tr(sym(grad(u)) * sym(grad(u)))

U = VectorFunctionSpace(mesh, "P", 1)
D = FunctionSpace(mesh, "DG", 0)
u, v = TrialFunction(U), TestFunction(U)
u_sol = Function(U)
density = Function(D, name="density")

# Define support
bcs = [DirichletBC(U, Constant((0.0, 0.0)), boundaries_mf, association_table["support"])]

# Define load
ds = Measure('ds', domain=mesh, subdomain_data=boundaries_mf)
F = dot(v, Constant((0.0, 1.0))) * ds(association_table["force"])

# Set up variational problem and solver
K = inner(density**penal * sigma(u), grad(v)) * dx
problem = LinearVariationalProblem(K, F, u_sol, bcs)
solver = LinearVariationalSolver(problem)

# Chage solver object to aboid UMFPack bug, more info:
# https://fenicsproject.org/qa/4177/reason-petsc-error-code-is-76/
# https://fenicsproject.org/pub/tutorial/html/._ftut1018.html
solver.parameters["linear_solver"] = "mumps"

# Set up neural network
top_optimizer = TopOptimizerNN(dim, neurons_per_layer=20, numbers_of_layers=10, use_softmax=True)
top_optimizer.apply(weight_init)

# Set up loss function
loss_function = TopOptLoss()

# Set up optimizer
optimizer = optim.Adam(top_optimizer.parameters(), amsgrad=True, lr=learning_rate)

# Calculate objective zero for scaling the loss function
density.vector()[:] = volume_ratio

solver.solve()

psi_0 = project(psi(u_sol), D).vector()[:].sum()
obj_0 = ((volume_ratio**penal) * psi_0).sum()

# Get mid points for each cell, filter and move them to cpu/gpu
mid_points = create_mid_points(mesh, dim)
mid_points = torch.tensor(mid_points, requires_grad=True).float().to(device)

# Get cell volumes
volumes = np.array([cell.volume() for cell in cells(mesh)])
volumes = torch.tensor(volumes, requires_grad=True).float().to(device)

vol_fraction = torch.sum(volumes) * volume_ratio

# Training data
training_data = []

# Training loop
for epoch in range(max_epochs):

    # Set gradients to zero
    optimizer.zero_grad()

    # Predict density for each cell
    density_new_tt = top_optimizer(mid_points)

    # Convert density to numpy array
    density_new_np = density_new_tt.cpu().detach().numpy()

    # Assign new density to function space and solve
    density.vector()[:] = density_new_np
    solver.solve()

    # Extract vector from function space
    psi_vector = project(psi(u_sol), D).vector()[:]

    # Objective function to minimize
    objective = torch.tensor((density_new_np**(2 * penal)) * psi_vector).float().to(device)

    # Calculate loss
    loss, vol_constraint = loss_function(density_new_tt, objective, vol_fraction, penal, obj_0,
                                         alpha, volumes)

    # Backward pass
    loss.backward(retain_graph=True)

    # Update alpha
    alpha = min(alpha_max, alpha + alpha_increment)

    # Apply gradient clipping, more info:
    # https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
    torch.nn.utils.clip_grad_norm_(top_optimizer.parameters(), nrm_threshold)

    optimizer.step()

    # Count grey elements
    grey_elements = sum(1 for rho in density_new_np if ((rho > 0.05) & (rho < 0.95)))
    rel_grey_elements = grey_elements / len(density_new_np)

    # Save density for the currect epoch
    xdmf.write(density, epoch)

    # Create log data
    objective_sum = float(objective.sum())
    density_avg = np.average(density_new_np)
    loss_item = loss.item()

    training_data.append([epoch, objective_sum, density_avg, loss_item, rel_grey_elements])

    # Print info
    if (epoch % log_interval == 0):
        print("{:3d} Objective: {:.2F}; Vf: {:.3F}; loss: {:.5F}; relGreyElems: {:.5F} ".format(
            epoch, objective_sum, density_avg, loss_item, rel_grey_elements))

    # When to stop training
    if ((epoch > min_epochs) & (rel_grey_elements < 0.035)):
        break

# Print time cost
stop = time.time()
elapsed = stop - start

elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed))
print("This optimization took:", elapsed)

# Save model
torch.save(top_optimizer.state_dict(), data_directory + "/model.pt")

# Save training data to file
write_to_csv(training_data, data_directory)

# Save optimization parameters to file
write_optimization_data(
    NAME, data_directory, {
        "Dimension": dim,
        "Learning rate": learning_rate,
        "Volume fraction": volume_ratio,
        "Penalty": penal,
        "Max epochs": max_epochs,
        "Min epochs": min_epochs,
        "Time": elapsed,
        "Epochs": epoch,
        "Clipping threshold": nrm_threshold
    })