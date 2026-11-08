"""
    Simplified 2D Heat Solidification Model of printed cross section
    Advanced Materials Team FY26
    260121 - OlivierXM

    Solve additive domains using DG elements of the first order
    Use a Nonlinear Solver to evaluate the evolution temperature
    Use a simple linear solidification model to mathematically adjust phase

    Proof of concept, only simulates individual blocks, not dynamic
    - Solves the steady state of the previous print
    - Adds next layer, sets initial state from previous steady state
    - Solves time evolution of new domain

    Boundaries
    - Ground is fixed at temperature
    - Side and top of structure exposed to convection and radiation
    - Assumed symmetry, domain split and adiabatic assumed

    Questions
    - Evaporative cooling?
    - Solar flux?
    - One sided exchange?
"""

# Standard Imports #
from mpi4py import MPI
from dolfinx import mesh, fem, nls, io, cpp, log
import dolfinx.nls.petsc
import dolfinx.fem.petsc
import numpy as np
import ufl
import petsc4py.PETSc as pets
import os
import sys
import datetime
import time
import math

# Custom Imports #
sys.path.append(sys.argv[1])
from dolfinxtools import cdfx_09 as cdfx 
from dolfinxtools import mpitools as mpt

# Load Arguments #
fullDir = sys.argv[3]
in2m = 0.0254
tzinfo = datetime.timezone(datetime.timedelta(hours = -5))
__version__ = "260130 - Individual layer solver"

# Inputs #
k_w = 0.56 # Thermal Conductivity of Water [W/mK]
k_s = 2.22 # Thermal Conductivity of Ice [W/mK]
rho_w = 999.9 # Density of Water [kg/m3]
rho_s = 916.2 # Density of Ice [kg/m3]
Cp_w = 4220 # Specific Heat of Water [J/kgK]
Cp_s = 2050 # Specific Heat of Ice [J/kgK]
latentHeat = 333400 # Latent Heat [J/kg]
meltTemp = 273 # Melting Temperature [K]
meltBounds = 0.25 # Mushy region half width [K]
convTemp = 273 - 10 # Ambient Temperature [K]
convH = 17.5 # Ambient Convection Coefficient [W/m2K]
groundTemp = 273 - 10 # Ground Temperature [K]
initialTemp = 273 + 3 # Initial mixture temperature [K]
emiss = 0.97 # Radiative emissivity [-]
stefanB = 5.67e-8 # Stefan Boltzmann constant [W/m2K4]

# Numerical Parameters #
alpha = 500 # Coercivity index [-]
dt1 = 0.2 # Timestep during thermal transience [s]
dt2 = 10 # Timestep during thermal stability [s]
dt3 = 10 # Timestep after solidification
rPrime = 0.0005 # Layer contact resistance [m2K/W]
layerHeight = 0.75 * in2m # Print Layer Height [m]
dX = (21.5 - 0* 0.75) * in2m # New layer length [m]
dX_2 = dX - layerHeight # Previous layer length [m]
dY = in2m # Half Width [m] //

freezeCriteria = 0.00025 # Stop when layer has reached [-]
saveFreq = 15 # Save every N seconds of sim time [s]
meltL = meltTemp - meltBounds
meltH = meltTemp + meltBounds
meltCstar = latentHeat / 2 / meltBounds
meltCderiv = (Cp_w - Cp_s) / 2 / meltBounds
# Generate Mesh #
tol = 1e-12
commWorld = MPI.COMM_WORLD
thisCore = commWorld.Get_rank()
nX = round(dX * (20/in2m)) # 12 cells/25.4mm
nY = round(dY * (20/in2m)) # 4 cells/25.4mm
domain = mesh.create_rectangle(commWorld, [[0,0],[dX,dY]], [nX,nY], cell_type = mesh.CellType.quadrilateral)
tDim = domain.topology.dim
domain.topology.create_connectivity(tDim - 1, tDim)
num_cells = domain.topology.index_map(tDim).size_local + domain.topology.index_map(tDim).num_ghosts
num_facets = domain.topology.index_map(tDim - 1).size_local + domain.topology.index_map(tDim - 1).num_ghosts
total_cells = commWorld.allreduce(num_cells, op = MPI.SUM)
# Create Functions/FunctionSpaces #
Fs_DG = fem.functionspace(domain, ("DG", 1))
Vs_DG = fem.functionspace(domain, ("DG", 1, (2,)))
u = fem.Function(Fs_DG, name = "Temperature")
u_n = fem.Function(Fs_DG)
phi = fem.Function(Fs_DG, name = "Phase")
phi_n = fem.Function(Fs_DG)
v = ufl.TestFunction(Fs_DG)

# Boundary Condition Markers #
class LeftSide(cdfx.SubDomain):
    def inside(self, x):
        arg1 = self.near(x[0], 0, tol)
        return arg1
    
class RightSide(cdfx.SubDomain):
    def inside(self, x):
        return self.near(x[0], dX, tol)
class TopSide(cdfx.SubDomain):
    def inside(self, x):
        return self.near(x[1], dY, tol)
    
class RightSide_Sub(cdfx.SubDomain):
    def inside(self, x):
        arg1 = self.near(x[0], dX_2, tol)
        arg2 = self.near(x[1], dY, tol)
        return self.lor(arg1, arg2)
    
class Boundary(cdfx.SubDomain):
    def inside(self, x):
        return np.isclose(x[0], dX_2, tol)
    
class Material(cdfx.SubDomain):
    def inside(self, x):
        return x[0] < dX_2 + tol

class NoneSide(cdfx.SubDomain):
    def inside(self, x):
        return abs(x[0]) < -1

# Facet Functions #
cellFunction = cdfx.FacetFunction(domain, tDim, num_cells)
Material().mark(cellFunction, 1)
cellTags = cellFunction.CreateMeshTag()
_ent = cellTags.find(0)

facetFunction = cdfx.FacetFunction(domain, tDim - 1, num_facets)
LeftSide().mark(facetFunction, 1)
RightSide().mark(facetFunction, 2)
TopSide().mark(facetFunction, 3)
facetTags = facetFunction.CreateMeshTag()

interiorFunction = cdfx.FacetFunction(domain, tDim - 1, num_facets)
Boundary().mark(interiorFunction, 1)
interiorTags = interiorFunction.CreateMeshTag()

# Boundary Conditions #
bcs_T = {1: {'Dirichlet': groundTemp, 'Surface' : 1, 'Marker' : LeftSide().inside},
         2: {'Robin': (convTemp, convH), 'Surface' : 2},
         3: {'Robin': (convTemp, convH), 'Surface' : 3},}


# Generate Sub Mesh #
sub_mesh, cell_map, vertex_map, geometry_map = mesh.create_submesh(domain, domain.topology.dim, cellTags.find(1)) # Map old volume to submesh
sub_mesh.topology.create_connectivity(sub_mesh.topology.dim - 1, sub_mesh.topology.dim)
Fs_sub = fem.functionspace(sub_mesh, ("DG", 1))
u_sub = fem.Function(Fs_sub)
v_sub = ufl.TestFunction(Fs_sub)
num_cells_sub = sub_mesh.topology.index_map(sub_mesh.topology.dim).size_local + sub_mesh.topology.index_map(sub_mesh.topology.dim).num_ghosts
num_sub_facets = sub_mesh.topology.index_map(sub_mesh.topology.dim - 1).size_local + sub_mesh.topology.index_map(sub_mesh.topology.dim - 1).num_ghosts

# Sub Mesh Markers #
facetFunction_sub = cdfx.FacetFunction(sub_mesh, sub_mesh.topology.dim - 1, num_sub_facets)
LeftSide().mark(facetFunction_sub, 1)
RightSide_Sub().mark(facetFunction_sub, 2)
facetTags_sub = facetFunction_sub.CreateMeshTag()

# Generate Measures #
dz = ufl.Measure('dx', domain = domain, subdomain_data = cellTags)
ds = ufl.Measure('ds', domain = domain, subdomain_data = facetTags)
dS = ufl.Measure('dS', domain = domain, subdomain_data = interiorTags)

dz_sub = ufl.Measure('dx', domain = sub_mesh)
ds_sub = ufl.Measure('ds', domain = sub_mesh, subdomain_data = facetTags_sub)
dS_sub = ufl.Measure('dS', domain = sub_mesh)

# Generate Boundary Terms #
bcs = cdfx.DirichletBCs(bcs_T, Fs_DG, facetTags, False)
bcs_sub = cdfx.DirichletBCs(bcs_T, Fs_sub, facetTags_sub, False)
robinBC = cdfx.RobinBCs(bcs_T, ds, u, v)
robinBC_sub = cdfx.RobinBCs(bcs_T, ds_sub, u_sub, v_sub)

# Get Interior Cell Parameters #
n = ufl.FacetNormal(domain)
n_sub = ufl.FacetNormal(sub_mesh)
h = ufl.CellDiameter(domain)
h_sub = ufl.CellDiameter(sub_mesh)

# Material Functions #
def p_eos(phi):
    """
        Convert phase to spline scale
        Args:
            phi: Phase, ufl object
    """
    return 6 * phi ** 5 - 15 * phi ** 4 + 10 * phi ** 3

def dp_deos(phi):
    """
        Calculate derivative of p_eos(phi)
        Args:
            phi: Phase, ufl object
    """
    return 30* phi ** 4 - 60 * phi ** 3 + 30 * phi ** 2

def kth(phi):
    return k_s*(1 - phi) + k_w*phi

def rho(phi):
    return rho_s*(1 - phi) + rho_w*phi

def Cp(phi):
    return Cp_s*(1 - phi) + Cp_w*phi


def GetCp(u_n, u):
    _phi_r = (u_n - meltL) / (2 * meltBounds)
    _phi = p_eos(_phi_r)
    return ufl.conditional(ufl.le(u_n, meltH), ufl.conditional(ufl.ge(u_n, meltL),Cp(_phi) + meltCstar + meltCderiv * dp_deos(_phi_r) * (u-meltL), Cp(0)), Cp(1))

# Interface Functions #
def AddInterface(N):
    """
        Add a DG interface term to the formulation
        Args:
            N: The interface marker
        Returns:
            The ufl compatible term
    """
    _F = alpha/ufl.avg(h)*ufl.dot(ufl.jump(v,n),rPrime*ufl.avg(kth(p_eos(phi_n)) * ufl.grad(u)))*dS(N)
    _F -= ufl.dot(ufl.avg(ufl.grad(v)), rPrime*ufl.avg(kth(p_eos(phi_n)) * ufl.grad(u)))*dS(N)
    _F -= ufl.dot(ufl.avg(ufl.grad(v)), ufl.jump(u, n))*dS(N)
    _F -= ufl.dot(ufl.avg(kth(p_eos(phi_n)) * ufl.grad(u)), ufl.jump(v, n))*dS(N)
    _F += alpha/ufl.avg(h)*ufl.dot(ufl.jump(v, n), ufl.jump(u, n))*dS(N)
    return _F

# Solve Sub Mesh Steady State #
F = (kth(p_eos(0)) * ufl.dot(ufl.grad(v_sub), ufl.grad(u_sub)))*dz_sub - robinBC_sub \
    - emiss * stefanB * (convTemp ** 4 - u_sub ** 4) *v_sub * ds_sub(2) -  emiss * stefanB * (convTemp ** 4 - u_sub ** 4)* v_sub *ds_sub(3) \
   + (-1*ufl.dot(ufl.avg(kth(p_eos(0)) * ufl.grad(v_sub)), ufl.jump(u_sub, n_sub))*dS_sub - ufl.dot(ufl.jump(v_sub, n_sub), ufl.avg(kth(p_eos(0)) * ufl.grad(u_sub)))*dS_sub + alpha/ufl.avg(h_sub)*ufl.dot(ufl.jump(v_sub, n_sub), ufl.jump(u_sub, n_sub))*dS_sub )

problem = dolfinx.fem.petsc.NonlinearProblem(F, u_sub, bcs = bcs_sub)
solver = dolfinx.nls.petsc.NewtonSolver(commWorld, problem)
solver.solve(u_sub)

# Establish Initial Conditions #
phi_n.x.array[:] = 0
for i in range(0, _ent.size):
    phi_n.x.array[Fs_DG.dofmap.cell_dofs(_ent[i])] = 1

u_n.x.array[:] = initialTemp
cdfx.transfer_submesh_data(u_n, u_sub, cell_map, True) # Map submesh temperature to parent
u_n.x.scatter_forward()
u.x.array[:] = u_n.x.array
phi_n.x.scatter_forward()

# - dt_sim*(k_w-k_s)*dp_deos(phi_n)*ufl.inner(ufl.grad(phi_n),ufl.grad(u_n))
# Establish Functional Form #
dt_sim = fem.Constant(domain, pets.ScalarType(dt1))
F = ((GetCp(u_n, u) * rho(p_eos(phi_n)) * (u - u_n) - dt_sim*(k_w-k_s)*dp_deos(phi_n)*ufl.inner(ufl.grad(phi_n),ufl.grad(u_n)))* v + dt_sim * kth(p_eos(phi_n)) * ufl.dot(ufl.grad(v), ufl.grad(u)))*dz - dt_sim * robinBC \
   - dt_sim * emiss * stefanB * ((convTemp ** 4 - u ** 4)*v * ds(2) + (convTemp ** 4 - u ** 4) * v * ds(3))  \
   + dt_sim * (-1*ufl.dot(ufl.avg(kth(p_eos(phi_n)) * ufl.grad(v)), ufl.jump(u, n))*dS(0) - ufl.dot(ufl.jump(v, n), ufl.avg(kth(p_eos(phi_n)) * ufl.grad(u)))*dS(0) + alpha/ufl.avg(h)*ufl.dot(ufl.jump(v, n), ufl.jump(u, n))*dS(0) )
   
F = F + dt_sim * AddInterface(1)

# Solvers #
problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs = bcs)
solver = dolfinx.nls.petsc.NewtonSolver(commWorld, problem)
solver.convergence_criterion = "residual"
solver.atol = 1e-6
solver.max_it = 25
solver.report = False

# Write Facet Files #
tempFile = io.VTKFile(commWorld, f"{fullDir}tempDir/temperature.pvd", "w")
phiFile = io.VTKFile(commWorld, f"{fullDir}phiDir/phi.pvd", "w")
facetFile = io.XDMFFile(commWorld, f"{fullDir}facets.xdmf", "w")

facetFile.write_mesh(domain)
facetFile.write_meshtags(facetTags, domain.geometry)  
facetFile.close()

cellFile = io.XDMFFile(commWorld, f"{fullDir}cells.xdmf", "w")
cellFile.write_mesh(domain)
cellFile.write_meshtags(cellTags, domain.geometry)  
cellFile.close()

# Write Details File #
specFile = mpt.MPIFile(commWorld, f"{fullDir}params.txt")
specFile.titleblock("Model Overview")
specFile.write(f"{datetime.datetime.now(tzinfo)}")
specFile.write(f"Script Name : {sys.argv[0]}")
specFile.write(f"Version : {__version__}")
specFile.write(f"Num Procs : {commWorld.Get_size()}")

specFile.titleblock("Numerical Overview", "\n")
specFile.write(f"Num Cells : {total_cells}")
specFile.write(f"Total Height : {dX:.2E} [m]")
specFile.write(f"New Layer Height : {layerHeight:.2E} [m]")
specFile.write(f"Initial Temperature : {initialTemp} [K]")
specFile.write(f"Ambient Convection h : {convH} [W/m2K]")
specFile.write(f"Ambient Convection T : {convTemp} [K]")
specFile.write(f"Surface Emissivity e : {emiss} [W/m2K4]")
specFile.write(f"Ground Temperature : {groundTemp} [K]")
specFile.write(f"Solidification Criteria : {freezeCriteria} [-]")
specFile.write(f"Initial time step : {dt1} [s]")
specFile.write(f"Secondary time step : {dt2} [s]")
specFile.write(f"Final time step : {dt3} [s]")

specFile.titleblock("Melt Kinetics", "\n")
specFile.write(f"Melt Temperature : {meltTemp} [K]")
specFile.write(f"Melt bounds : {meltBounds} [K]")
specFile.write(f"Contact Resistance : {rPrime} [m2K/W]")
specFile.close()


# Initialize Simulation #
tempFile.write([u_n._cpp_object], 0)
phiFile.write([phi_n._cpp_object], 0)
outFile = mpt.MPIFile(commWorld, f"{fullDir}data.txt")
t = 0
saveT = 0
t0 = time.time()
dt = dt1
tStop = 25000
callOnce = True
# Simulate #
while (t < tStop):
    # Timestep #
    t += dt
    saveT += dt

    # Solve Temperature #
    solver.solve(u)

    # Solve Phase #
    phi.x.array[:] = (u.x.array - meltL) / (2 * meltBounds)

    # Assign Previous Step #
    phi.x.array[:] = np.clip(phi.x.array, 0, 1) # Clip phase to [0, 1]
    phi_n.x.array[:] = phi.x.array
    u_n.x.array[:] = u.x.array
    u_n.x.scatter_forward()
    phi_n.x.scatter_forward()
    # Save every few timesteps #
    if (saveT > saveFreq):
        saveT = 0
        tempFile.write([u._cpp_object], t)
        phiFile.write([phi._cpp_object], t)
        _vol = commWorld.allreduce(fem.assemble_scalar(fem.form(phi * dz(0))), op = MPI.SUM) / dY / layerHeight
        _surfTemp = commWorld.allreduce(fem.assemble_scalar(fem.form(u * ds(2))), op = MPI.SUM) / dY
        mpt.MPIprint(thisCore, 0, f"{t:.1f}s ({t/3600:.1f}h) : {_vol:0.7f} : {(time.time()-t0)/60:0.1f}min")
        outFile.write(f"{t:.5f}",feed = ",")
        outFile.write(f"{_vol:0.7f}",feed = ",")
        outFile.write(f"{_surfTemp:0.2f}")

        # Upper stable regime check #
        if (_vol < 0.95 and callOnce):
            dt_sim.value = dt2
            dt = dt2

        # Lower stable regime check #
        if (_vol < 0.01 and callOnce):
            dt_sim.value = dt1
            dt = dt1

        # Exit upon reaching criteria #
        if (_vol < freezeCriteria and callOnce):
            dt_sim.value = dt3
            dt = dt3
            callOnce = False
            tStop = t + 4800 # 4800
            # break

# Close Files #
tempFile.close()
phiFile.close()
outFile.close()
# End Script #