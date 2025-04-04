"""
    Background Grid with a center grid on top
"""

#### PHYSICAL CONSTANTS ###
BOLTZMANN_IN_ERG_PER_KELVIN = 1.380649e-16
Hydrogen_mass_in_g = 1.6735e-24 # 1 hydrogen mass in grams
gamma = 5.0/3.0

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
boxsize = 100 # Units in kiloparsecs 
midpoint = boxsize/2

## Position of first and last cell in the background
background_cells_per_dimension = 5
dx_bg = boxsize/background_cells_per_dimension
pos_first, pos_last = 0.5*dx_bg, boxsize - 0.5*dx_bg # faces of the cell are placed in between the borders

center_len = 5
center_beginning = midpoint - center_len
center_end = midpoint + center_len
center_cells_per_dimension = 51
dx_center = (center_end - center_beginning)/center_cells_per_dimension
pos_first_center, pos_last_center = 0.5*dx_center + center_beginning, center_end - 0.5*dx_center

# Fill with background values
density_0 = 1e-4 # hydrogem atoms/cm^3 
velocity_radial_0 = 0 # initial radial velocity - in km/s

# T = (gamma - 1)*u/kb*UnitEnergy/UnitMass*mu, => T = (gamma - 1)* u/(1.380649e-16 ergs/Kelvin) * (1.988e43 ergs/1.989e33 grams) *  (~0.6*1.6727e-24g)
# T[K] = 2/3* u(Code Units) * 68.48 Kelvin. Where u = (u_therm_0)/(UnitEnergy_in_cgs)/(density_code*dx^3) or u = pressure_0/(gamma - 1.0)/density_0/(density_code*dx^3) 
# T[K] = pressure_0/density_0/(density_code*dx^3)/kb*UnitEnergy/UnitMass*mu
# T[K] = pressure_0/(density_0*density_code*dx^3*kb)   *    (UnitEnergy/UnitMass*mu)
# T[K]/mu*UnitMass/UnitEnergy*(density_0*density_code*kb) = 

T = 1e6 # Preferred background temperature in Kelvin
# T = (gamma - 1)*u/kb*UnitEnergy/UnitMass*mu => 
# (UnitMass_in_g)*(BOLTZMANN_IN_ERG_PER_KELVIN/(0.63*Hydrogen_mass_in_g)*T  = (pressure_0/density_0))/(density_code*pow(dx,3))

p0_bg = T*(dx_bg**3)*(density_0**2)/(UnitDensity_in_cgs)*BOLTZMANN_IN_ERG_PER_KELVIN*UnitMass_in_g/0.63  # Per my derivation, this corresponds to P = 3/2 N^2 kb * T, where N is the number of hydrogen atoms(N^2 is rather strange)
p0_center = T*(dx_center**3)*(density_0**2)/(UnitDensity_in_cgs)*BOLTZMANN_IN_ERG_PER_KELVIN*UnitMass_in_g/0.63  # Per my derivation, this corresponds to P = 3/2 N^2 kb * T, where N is the number of hydrogen atoms(N^2 is rather strange)


#### DON'T CHANGE THESE ####
u_therm_bg = p0_bg/(gamma - 1.0)/density_0 
u_therm_center = p0_center/(gamma - 1.0)/density_0 

# Convert density u_therm_0, and pressure to code units
density_code = (density_0*Hydrogen_mass_in_g)/(UnitDensity_in_cgs)
energy_code_bg = (u_therm_bg)/(UnitEnergy_in_cgs) 
energy_code_center = (u_therm_center)/(UnitEnergy_in_cgs) 

# set up the values for one dimension
grid_1d = np.linspace(pos_first, pos_last, background_cells_per_dimension)
background_number_of_cells = pow(background_cells_per_dimension, 3) 
xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d)

# set up the grid as seen in many Arepo examples
bg_pos = np.zeros([background_number_of_cells, 3]) 
bg_pos[:,0] = xx.flatten()
bg_pos[:,1] = yy.flatten()
bg_pos[:,2] = zz.flatten()
center = np.where(np.all(bg_pos == [50,50,50], axis=1))[0] # replace the center with the central disk,
bg_pos = np.delete(bg_pos, center, axis=0)
background_number_of_cells -= 1

bg_mass = np.zeros(background_number_of_cells) 
bg_energy = np.zeros(background_number_of_cells)

volume_code_bg = pow(dx_bg,3)
mass_code_bg = density_code*volume_code_bg
energy_code_specific_bg = energy_code_bg/mass_code_bg
bg_mass = np.add(bg_mass, mass_code_bg)
bg_energy = np.add(bg_energy, energy_code_specific_bg)

# Set up the center grid
grid_1d_center = np.linspace(pos_first_center, pos_last_center, center_cells_per_dimension)
print("beginning of center region", center_beginning)
print("end of center region", center_end)
print(len(grid_1d_center))

center_number_of_cells = pow(len(grid_1d_center),3) 

xx_c, yy_c, zz_c = np.meshgrid(grid_1d_center, grid_1d_center, grid_1d_center)
center_pos = np.zeros([center_number_of_cells, 3]) 
center_pos[:,0] = xx_c.flatten()
center_pos[:,1] = yy_c.flatten()
center_pos[:,2] = zz_c.flatten()

center_mass = np.zeros(center_number_of_cells) 
center_energy = np.zeros(center_number_of_cells)

volume_code_center = pow(dx_center,3)
mass_code_center = density_code*volume_code_center
energy_code_specific_center = energy_code_center/mass_code_center

center_mass = np.add(center_mass, mass_code_center)
center_energy = np.add(center_energy, energy_code_specific_center)

pos = np.vstack([bg_pos, center_pos]) 
number_of_cells = len(pos)

# Add mass and energy to the initial arrays
Mass = np.concatenate([bg_mass, center_mass])
Energy = np.concatenate([bg_energy, center_energy])

Velocity = np.zeros([number_of_cells, 3])
unique_coordinates = np.unique(pos, axis=0)

# import pdb; pdb.set_trace()
"""write *.hdf5 file; minimum number of fields required by Arepo """
IC = h5py.File(simulation_directory + 'background_grid_100_100_150x150.hdf5', 'w')

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