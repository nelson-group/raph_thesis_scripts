"""
    Plots out an "analytic" disk profile for the gas density in the disk.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


#### PHYSICAL CONSTANTS ###
BOLTZMANN_IN_ERG_PER_KELVIN = 1.380649e-16
Hydrogen_mass_in_g = 1.6735e-24 # 1 hydrogen mass in grams
GRAVITIONAL_CONSTANT_IN_CGS = 6.6738e-8

#### SIMULATION CONSTANTS - keep it consistent with param.txt ###
UnitVelocity_in_cm_per_s = 1e5 # 1 km/sec 
UnitLength_in_cm = 3.085678e21 # 1 kpc
UnitMass_in_g = 1.989e33 # 1 solar mass
UnitDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**3 # 6.769911178294545e-32 g/cm^3
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.988e43 ergs
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye

#### load libraries
import sys    # system specific calls
import numpy as np    ## load numpy
import h5py    ## load h5py; needed to write initial conditions in hdf5 format
from scipy.integrate import quad

# print("thesis/cc85/create.py: creating ICs in directory" +  simulation_directory)

""" Initial Condition Parameters """
# FilePath = simulation_directory + 'IC300kpc.hdf5'

# Setting up the box, modify the values as needed
boxsize = 30 # Units in kiloparsecs 
cells_per_dimension = 100 # resolution of simulation simulation 
number_of_cells = pow(cells_per_dimension, 3) 

# Fill with background values
density_0 = 1e-4 # hydrogem atoms/cm^3 
velocity_radial_0 = 0 # initial radial velocity - in km/s
pressure_0 = 5e44 # K/cm^3
pressure_cgs = pressure_0*BOLTZMANN_IN_ERG_PER_KELVIN # erg/cm^3

#### DON'T CHANGE THESE ####
gamma = 5.0/3.0
u_therm_0 = pressure_0/(gamma - 1.0)/density_0 # this is in terms of erg
# Set up the grid
dx = boxsize/cells_per_dimension # code units
## Position of first and last cell
pos_first, pos_last = 0.5*dx, boxsize - 0.5*dx # faces of the cell are placed in between the borders

#### DISK - Values are taken     from Schneider and Robertson 2018("Introducing CGOLS") #### 
Mstars = 1e10 # Mass of the stellar disk = 1e10 solar masses
Rstars = 0.8 # Stellar scale radius = 0.8 kpc 
Mgas = 2.5e9 # total gas mass = 2.5e9 solar masses
zstars_in_UnitLength = 0.15 # 0.15 kpc
Rgas = Rstars*2 # disk scale length = 1.6 kpc 
central_surface_density = (Mgas)/(2*np.pi*pow(Rgas,2)) # Mgas = (2.5e9 * 1.989e33 grams)/(2*3.14*(1.6*3.085678e21)^2 cm^2) = 0.03248 g/cm^2
csd = (Mgas*UnitMass_in_g)/(2*np.pi*pow(Rgas*UnitLength_in_cm,2)) # Mgas = (2.5e9 * 1.989e33 grams)/(2*3.14*(1.6*3.085678e21)^2 cm^2) = 

DMMass = 5e10 # Mhalo or Mvir = 5e10 solar masses
halo_concentration = 10 # halo concentration
Rvir = 53 # virial radius = 53 kpc 
T_disk_in_Kelvin = 1e4 # Disk temperature in kelvin
mean_molecular_weight = 0.6 # mean molecular weight
Rhalo = Rvir/halo_concentration # Scale radius of the halo, given by Rvir/c 
halo_sound_speed = np.sqrt(BOLTZMANN_IN_ERG_PER_KELVIN*T_disk_in_Kelvin/(mean_molecular_weight*Hydrogen_mass_in_g))/UnitVelocity_in_cm_per_s # cm/s / (UnitVelocity_in_cm_per_s)

fig = plt.figure(figsize=(20,12))

# Make unit conversions first -> then rewrite the potentials 
# Potentials: Total Potential = stellar_halo_potential + NFW_DM_halo_potential
# Miyamoto-Nagai Profile: Physical Units
def stellar_halo_potential(z, radial): # radial and z are the radial and vertical cylindrical coordinates
    # Use G in code units, then unit conversions are unnecessary
    G_code = GRAVITIONAL_CONSTANT_IN_CGS/(pow(UnitLength_in_cm,3) * pow(UnitMass_in_g, -1) * pow(UnitTime_in_s, -2))
    return -G_code*(Mstars)/(np.sqrt(pow(radial,2) + pow(Rstars + np.sqrt(pow(z, 2) + pow(zstars_in_UnitLength,2)) , 2)))

# NFW Profile: Physical Units
def NFW_DM_halo_potential(radius): # radius is the radius in spherical coordinates.
    # Use G in code units, then unit conversions are unnecessary
    G_code = GRAVITIONAL_CONSTANT_IN_CGS/(pow(UnitLength_in_cm,3) * pow(UnitMass_in_g, -1) * pow(UnitTime_in_s, -2))    
    return -G_code * DMMass/(radius*(np.log(1+halo_concentration) - halo_concentration/(1 + halo_concentration)))*np.log(1+(radius)/(Rhalo))

# Disk Gas 
def radial_gas_distribution(radial):
    return central_surface_density*np.exp(-(radial)/(Rgas)) # NOTE: this is in cgs units, not code ones

# Set the velocities with:
def acceleration(r,z): #place holder. Meant for calculating velocity.
    return 0

# Halo Gas 
initial_halo_gas_density_100_kpc = 3e3 * UnitMass_in_g/pow(UnitLength_in_cm, 3)
def halo_gas(radial, z, radius):
    full_halo = stellar_halo_potential(radial,z) + NFW_DM_halo_potential(radius)
    mid_point_halo = stellar_halo_potential(0,0) + NFW_DM_halo_potential(0) # placeholder value
    return initial_halo_gas_density_100_kpc*(1+(gamma - 1))*(full_halo - mid_point_halo)/halo_sound_speed

# set up the values for one dimension
grid_1d = np.linspace(pos_first, pos_last, cells_per_dimension)
xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d)

# set up the grid as seen in many Arepo examples
pos = np.zeros([number_of_cells, 3]) 
# Adds randomization to the cell positions - dis/enable as needed
# rng = np.random.default_rng(314159) 
# del_xyz = np.random.uniform(low=-1e-5, high=1e-5, size=(number_of_cells, 3))# to prevent a uniform grid
# pos += del_xyz

pos[:,0] = xx.flatten()
pos[:,1] = yy.flatten()
pos[:,2] = zz.flatten()

"""set up hydrodynamical quantities"""
Mass = np.zeros(number_of_cells) 
Energy = np.zeros(number_of_cells)
Velocity = np.zeros([number_of_cells, 3])

"""fill with code units"""
# Convert density u_therm_0, and pressure to code units
density_code = (density_0*Hydrogen_mass_in_g)/(UnitDensity_in_cgs)
energy_code = (u_therm_0)/(UnitEnergy_in_cgs) 
volume_code = pow(dx,3)
mass_code = density_code*volume_code
energy_code_specific = energy_code/mass_code
# Add mass and energy to the initial arrays
Energy = np.add(Energy, energy_code_specific)
Mass = np.add(Mass, mass_code) 


rad_x = pos[:,0] - 0.5*boxsize
rad_y = pos[:,1] - 0.5*boxsize
rad_z = pos[:,2] - 0.5*boxsize
rad_xy = np.sqrt(rad_x**2 + rad_y**2)
radii = np.sqrt(rad_x**2 + rad_y**2 + rad_z**2)

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

ax1 = fig.add_subplot(1,2,1)
ax1.plot(radii,radial_gas_distribution(radii))
ax1.set_xlabel("Distance [kpc]")
ax1.set_ylabel("Surface Density [g/cm]")
       
ax1.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

# DISK
for i, radius in enumerate(radii):
     # Insert disk gas within a cylindical region of diameter 3*(Disk Scale) and smallest z values 
    if rad_xy[i] <= Rgas*3/2: # if rad_xy <= are in a cylindrical region of space
        surface_density = radial_gas_distribution(rad_xy[i]) # radial distribution of density

        ### Vertical height profile for a given radius ###
        halo_potential = stellar_halo_potential(rad_z[i], rad_xy[i]) + NFW_DM_halo_potential(radius) 

        mid_plane_potential = stellar_halo_potential(0, rad_xy[i]) +  NFW_DM_halo_potential(radius) 

        def f(z):
            return np.exp(- ((stellar_halo_potential(z, rad_xy[i]) + NFW_DM_halo_potential(radius)) - mid_plane_potential)/halo_sound_speed**2)

        mid_plane_density = surface_density/quad(f, np.min(rad_z), np.max(rad_z)) 

        density = mid_plane_density*np.exp(-(halo_potential - mid_plane_potential)/halo_sound_speed**2) # equation 4 
        Mass[i] += density[0]*pow(dx,3)

plt.savefig("analytic" + ".png", dpi=150, bbox_inches='tight') 
plt.show()