"""
    Initial conditions for the CC85 model.

    Based on the Noh 3D example in the public version of Arepo. 

    Physical Situation: A constant* source of mass and energy is deposited in a spherical volume of radius R 
    Characterized by 3 key parameters:
    - Mass Injection Rate: M_wind = Beta*M_sfr for mass loading factor Beta
    - Energy Injection Rate: E_wind = alpha*E_sn  for energy loading factor
    - Radius of Injection Region: R
    - If each supernova releases 10e41 ergs and there is 1 supernova per 100 solar masses: [E_wind = 3e41*alpha*M_wind]
"""

#### PHYSICAL CONSTANTS ###
BOLTZMANN_IN_ERG_PER_KELVIN = 1.380649e-16
Hydrogen_mass_in_g = 1.6735e-24 # 1 hydrogen mass in grams

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

simulation_directory = str(sys.argv[1])
print("thesis/cc85/create.py: creating ICs in directory" +  simulation_directory)

""" Initial Condition Parameters """
FilePath = simulation_directory + 'IC150kpc.hdf5'

# Setting up the box, modify the values as needed
boxsize = 30 # Units in kiloparsecs 
cells_per_dimension = 450 # resolution of simulation simulation 
number_of_cells = pow(cells_per_dimension, 3) 

# Fill with background values
density_0 = 1e-4 # hydrogem atoms/cm^3 
velocity_radial_0 = 0 # initial radial velocity - in km/s
# Figure out the CGS units of pressure. 
pressure_0 = 3e43
pressure_cgs = pressure_0*BOLTZMANN_IN_ERG_PER_KELVIN 

#### DON'T CHANGE THESE ####
gamma = 5.0/3.0
u_therm_0 = pressure_0/(gamma - 1.0)/density_0 # 

# Set up the grid
dx = boxsize/cells_per_dimension # code units
## Position of first and last cell
pos_first, pos_last = 0.5*dx, boxsize - 0.5*dx # faces of the cell are placed in between the borders

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
Mass = np.add(Mass, mass_code) 
Energy = np.add(Energy, energy_code_specific)

# temperature calculation - taken from illustris tng
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS # ~0.63*proton mass
def Temp_S(x_e,ie):
    return (gamma - 1) * energy_code/BOLTZMANN_IN_ERG_PER_KELVIN * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)
print("temperature", Temp_S(1, energy_code_specific))


"""write *.hdf5 file; minimum number of fields required by Arepo """
IC = h5py.File(simulation_directory + 'IC30kpc_450.hdf5', 'w')

## create hdf5 groups
header = IC.create_group("Header")
part0 = IC.create_group("PartType0")
## header entries
NumPart = np.array([number_of_cells, 0, 0, 0, 0, 0])
header.attrs.create("NumPart_ThisFile", NumPart)
header.attrs.create("NumPart_Total", NumPart)
header.attrs.create("NumPart_Total_HighWord", np.zeros(6) )
header.attrs.create("MassTable", np.zeros(6) )
header.attrs.create("Time", 0.0)
header.attrs.create("Redshift", 0.0)
header.attrs.create("BoxSize", boxsize)
header.attrs.create("NumFilesPerSnapshot", 1)
header.attrs.create("Omega0", 0.0)
header.attrs.create("OmegaB", 0.0)
header.attrs.create("OmegaLambda", 0.0)
header.attrs.create("HubbleParam", 1.0)
header.attrs.create("Flag_Sfr", 0)
header.attrs.create("Flag_Cooling", 0)
header.attrs.create("Flag_StellarAge", 0)
header.attrs.create("Flag_Metals", 0)
header.attrs.create("Flag_Feedback", 0)
if pos.dtype == np.float64:
    header.attrs.create("Flag_DoublePrecision", 1)
else:
    header.attrs.create("Flag_DoublePrecision", 0)

## copy datasets
part0.create_dataset("ParticleIDs", data = np.arange(1, number_of_cells+1) )
part0.create_dataset("Coordinates", data = pos)
part0.create_dataset("Masses", data = Mass)
part0.create_dataset("Velocities", data = Velocity)
part0.create_dataset("InternalEnergy", data = Energy)

## close file
IC.close()