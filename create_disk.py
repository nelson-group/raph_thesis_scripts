"""
    Initial conditions for the CC85 model.

    Based on the Noh 3D example in the public version of Arepo. 

    Physical Situation: A constant* source of mass and energy is deposited in a spherical volume of radius R 
    Characterized by 3 key parameters:
    - Mass Injection Rate: M_wind = Beta*M_sfr for mass loading factor Beta
    - Energy Injection Rate: E_wind = alpha*E_sn  for energy loading factor
    - Radius of Injection Region: R
    - If each supernova releases 10e41 ergs and there is 1 supernova per 100 solar masses: [E_wind = 3e41*alpha*M_wind]

    The disk profiles and values are taken from: Introducing CGOLS: the Cholla galactic outfLow simulation suite by Schneider and Robertson
"""

###################################
###################################
# CONFIGURATION OPTIONS #
## General Cofiguration Options ##
RANDOMIZATION = False # Adds randomization to the cell positions. Arepo and the Riemann solver can occassionaly be stressed out by a random grid
BACKGROUND_BOX = True # Creates a box that surrounds the grid. 
BG_BOXSIZE = 100 # Size of the outer layer. Should be greater than the boxsize -> Currrently unused and I just have a box
OUTPUT_CSV = False # creates csv files for certain values
OUTPUT_DEBUG_PLOTS = True # creates debu files for certain values

## Potentials that we need for the disk ##
STELLAR_POTENTIAL = True # Configuration option for the stellar potential
NFW_POTENTIAL = False # Configuration option for the NFW potential - Calls CGOL's NFW 
NFW_POTENTIALV2 = False # Configuration for the disk - Calls Arepo's NFW.
###################################
###################################

#### PHYSICAL CONSTANTS ###
BOLTZMANN_IN_ERG_PER_KELVIN = 1.380649e-16
Hydrogen_mass_in_g = 1.6735e-24 # 1 hydrogen mass in grams
GRAVITIONAL_CONSTANT_IN_CGS = 6.6738e-8
HUBBLE = 3.2407789e-18
M_PI = 3.14159265358979323846

#### SIMULATION CONSTANTS - keep it consistent with param.txt ###
UnitVelocity_in_cm_per_s = 1e5 # 1 km/sec 
UnitLength_in_cm = 3.085678e21 # 1 kpc
UnitMass_in_g = 1.989e33 # 1 solar mass
UnitDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**3 #  1.989e33g/(3.085678e21 cm)^3 = 6.769911178294545e-32 g/cm^3
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.085678e21 cm / 10000cm/s = 3.08e16 seconds
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) #  1.989e33 grams*(3.085678e21cm)^2/(3.08e16 seconds)^2 = 1.988e43 grams*cm^2/second^2 :ergs
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 1.989e33 grams/3.085678e21cm/(3.08e16 seconds)^2 = 6.769911178294542e-22: (grams/(cm second^2))
UnitNumberDensity = UnitDensity_in_cgs/Hydrogen_mass_in_g
UnitSurfaceDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**2
UnitSurfaceNumberDensity = UnitSurfaceDensity_in_cgs/Hydrogen_mass_in_g
#### load libraries
import sys    # system specific calls
import numpy as np    ## load numpy
import h5py    ## load h5py; needed to write initial conditions in hdf5 format
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy import stats
import time

start = time.time()
simulation_directory = str(sys.argv[1])
print("thesis/cc85/create.py: creating ICs in directory" +  simulation_directory)

""" Initial Condition Parameters """
# Setting up the box, modify the values as needed
# boxsize = 10 kpc for m82 
boxsize = 10 # Units in kiloparsecs 
# cells per dim -> 301 kpc for m82
cells_per_dimension = 301 # resolution of simulation  # 0.03322259136212625 # WARNING THIS IS GOING TO BE V

number_of_cells = pow(cells_per_dimension, 3) 
dx = boxsize/cells_per_dimension # code units
# print(dx)
# Fill with background values
density_0 = 0.0615/1000  # hydrogem atoms/cm^3 
# print(density_0)
# G in code units is: UnitLength^3/(UnitMass*UnitTime^2). CGS equivalent: (g*cm/s^2)*(cm^2*g^-2) or cm^3/(grams*s)
G_code = GRAVITIONAL_CONSTANT_IN_CGS/(pow(UnitLength_in_cm,3) * pow(UnitMass_in_g, -1) * pow(UnitTime_in_s, -2)) 

velocity_radial_0 = 0 # initial radial velocity - in km/s

# T = (gamma - 1)*u/kb*UnitEnergy/UnitMass*mu, => T = (gamma - 1)* u/(1.380649e-16 ergs/Kelvin) * (1.988e43 ergs/1.989e33 grams) *  (~0.6*1.6727e-24g)
# T[K] = 2/3* u(Code Units) * 68.48 Kelvin. Where u = (u_therm_0)/(UnitEnergy_in_cgs)/(density_code*dx^3) or u = pressure_0/(gamma - 1.0)/density_0/(density_code*dx^3) 
# T[K] = pressure_0/density_0/(density_code*dx^3)/kb*UnitEnergy/UnitMass*mu
# T[K] = pressure_0/(density_0*density_code*dx^3*kb)   *    (UnitEnergy/UnitMass*mu)
# T[K]/mu*UnitMass/UnitEnergy*(density_0*density_code*kb) = 
# Background_T = 1e6 # Background temperature for m82
Background_T = 1.0e6 # Background temperature of the milky way
# T = (gamma - 1)*u/kb*UnitEnergy/UnitMass*mu => 
# (UnitMass_in_g)*(BOLTZMANN_IN_ERG_PER_KELVIN/(0.63*Hydrogen_mass_in_g)*T  = (pressure_0/density_0))/(density_code*pow(dx,3))
pressure_0 = Background_T*(dx**3)*(density_0**2)/(UnitDensity_in_cgs)*BOLTZMANN_IN_ERG_PER_KELVIN*UnitMass_in_g/(0.63)  

#### DON'T CHANGE THESE ####
gamma = 5.0/3.0
u_therm_0 = pressure_0/(gamma - 1.0)/density_0 # Another aspect of the ideal gas is the equation of state relating the pressure to the internal specific energy e
# Set up the grid
## Position of first and last cell
pos_first, pos_last = 0.5*dx, boxsize - 0.5*dx # faces of the cell are placed in between the borders

## DISK - Based on the Miyamoto-Nagai Disk as presented in Schneider et. al ##
Mstars = 1e10 # Mass of the stellar disk = 1e10 solar masses for m82
Rstars = 0.8 # Stellar scale radius = 0.8 kpc for m82 
Mgas = 2.5e9 # total gas mass = 2.5e9 solar masses for m82
zstars_in_UnitLength = 0.15 # 0.15 kpc
Rgas = Rstars*2 # disk scale length = 1.6 kpc 
central_surface_density = Mgas/(2*np.pi*pow(Rgas,2)) # UnitMass/UnitLength**2 = UnitSurfaceVolume
NFW_M200 = 5e10 # Mhalo or Mvir = 5e10 solar masses
NFW_C = 10 # concentration of the NFW potential, 10 for M82
Rvir = 53 # virial radius = 53 kpc for m82

T_disk_in_Kelvin = 1e4 # Disk temperature in kelvin. Note that altering the disk temperature affects the flaring in the disk. 1e3K has no flaring compared to 1e4K
mean_molecular_weight = 0.6 # mean molecular weight
Rhalo = Rvir/NFW_C # Scale radius of the halo, given by Rvir/c 
disk_sound_speed = np.sqrt(BOLTZMANN_IN_ERG_PER_KELVIN*T_disk_in_Kelvin/(mean_molecular_weight*Hydrogen_mass_in_g))/UnitVelocity_in_cm_per_s # g * cm^2/s^2 * /K * K / g = sqrt(cm^2/s^2)/UnitVelocity_in_cm_per_s. The sound speed affects how much flaring the disk will have. A higher disk sound speed = more flaring

## Options for NFWV2
if (NFW_POTENTIALV2): # NFW_PotentialV2 requires a lot of variables..
    NFW_Eps = 0.05
    Hubble_code = HUBBLE * UnitTime_in_s # All.Hubble in Arepo
    RhoCrit = 3 * Hubble_code * Hubble_code / (8 * M_PI * G_code) 
    print("RhoCrit", RhoCrit) # Arepo's value RhoCrit: 277.475116, last printed value: 277.4751161473195
    R200 = pow( NFW_M200 * G_code / (100 * Hubble_code * Hubble_code), 1.0 / 3) 
    print("R200", R200) # Arepo's value R200: 59.915950, last printed value: 59.91595038920018
    V200    = 10 * Hubble_code * R200 
    print("V200", V200) # Arepo's value V200: 59.916, Last printed value 59.91595131546602
    Rs = R200 / NFW_C
    Dc = 200.0 / 3 * (NFW_C * NFW_C * NFW_C) / (np.log(1 + NFW_C) - NFW_C/(1 + NFW_C))
    print("NFW_C ", NFW_C)
    print("log(1 + NFW_C) ", np.log(1 + NFW_C))
    print("NFW_C / (1 + NFW_C) ", NFW_C / (1 + NFW_C))
    print("Denominator", (np.log(1 + NFW_C) - NFW_C/(1 + NFW_C)))
    print("Dc ", Dc)
    fac = 1
    def enclosed_mass(R):
        ## Eps is in units of Rs !!!! ## 
        R = np.minimum(R, Rs * NFW_C)
        return fac * 4 * M_PI * RhoCrit * Dc * (-(Rs * Rs * Rs * (1 - NFW_Eps + np.log(Rs) - 2 * NFW_Eps * np.log(Rs) + NFW_Eps * NFW_Eps * np.log(NFW_Eps * Rs))) / ((NFW_Eps - 1) * (NFW_Eps - 1)) + (Rs * Rs * Rs * (Rs - NFW_Eps * Rs - (2 * NFW_Eps - 1) * (R + Rs) * np.log(R + Rs) + NFW_Eps * NFW_Eps * (R + Rs) * np.log(R + NFW_Eps * Rs))) /((NFW_Eps - 1) * (NFW_Eps - 1) * (R + Rs)))

    Mtot = enclosed_mass(R200)
    print("Mtot", Mtot)
    fac  = V200 * V200 * V200 / (10 * G_code * Hubble_code) / Mtot
    print(fac)
    Mtot = enclosed_mass(R200)
    print("Mtot", Mtot)
###################################
###################################

# Miyamoto-Nagai profile for the galaxy’s stellar disk
# The (CYLINDRICAL)potential is in code units: UnitLength^3*UnitMass/(UnitMass * UnitTime^2)/sqrt(UnitLength^2) = UnitLength^2/UnitTime^2. - CGS equivalent: cm^2/s^2
def stellar_potential(z, radial): 
    return -G_code*(Mstars)/(np.sqrt(pow(radial,2) + pow(Rstars + np.sqrt(pow(z, 2) + pow(zstars_in_UnitLength,2)) , 2)))

# NFW Profile to represent the halo potential
# THE (SPHERICAL)potential is:  UnitLength^3/(UnitMass*UnitTime^2)/UnitMass = UnitLength^2/(UnitTime^2) - CGS equivalent: cm^2/s^2
def NFW_DM_halo_potential(radius): # radius is the radius in spherical coordinates.
    return -G_code * NFW_M200/(radius*(np.log(1 + NFW_C) - NFW_C/(1 + NFW_C)))*np.log(1+(radius)/(Rhalo))

def NFW_DM_halo_potential_v2(radius):
    m = enclosed_mass(radius)
    return -(G_code * m) / radius 

# Disk Gas 
# Units are in UnitDensity
def radial_gas_distribution(radial):
    return central_surface_density*np.exp(-(radial)/(Rgas))
if (NFW_POTENTIALV2) and (NFW_POTENTIAL):
    raise Exception("Both NFW_POTENTIAL and NFW_POTENTIALV2 are enabled. Please disable one and run again.")
if (STELLAR_POTENTIAL) and (NFW_POTENTIAL):
    print("STELLAR_POTENTIAL and NFW_POTENTIAL enabled")
elif (STELLAR_POTENTIAL) and (NFW_POTENTIALV2):
    print("STELLAR_POTENTIAL and NFW_POTENTIALV2 enabled")
elif (STELLAR_POTENTIAL):
    print("STELLAR_POTENTIAL enabled")
elif (NFW_POTENTIAL):
    print("NFW_POTENTIAL enabled")
elif (NFW_POTENTIALV2):
    print("NFW_POTENTIALV2 enabled")
# Define a new function called total potential that takes in all radz, rad_xy and radii and returns to the summation of the potentials
def total_potential(z, radial, radius):
    if (STELLAR_POTENTIAL) and (NFW_POTENTIAL):
        return stellar_potential(z, radial) + NFW_DM_halo_potential(radius)
    elif (STELLAR_POTENTIAL) and (NFW_POTENTIALV2):
        return stellar_potential(z, radial) + NFW_DM_halo_potential_v2(radius)
    elif (STELLAR_POTENTIAL):
        return stellar_potential(z, radial)
    elif (NFW_POTENTIAL):
        return NFW_DM_halo_potential(radius)
    elif (NFW_POTENTIALV2):
        return NFW_DM_halo_potential_v2(radius)

# set up the values for one dimension
grid_1d = np.linspace(pos_first, pos_last, cells_per_dimension) 
xx, yy, zz = np.meshgrid(grid_1d, grid_1d, grid_1d)
pos = np.zeros([number_of_cells, 3]) 
if (RANDOMIZATION):
    rng = np.random.default_rng(314159) 
    del_xyz = np.random.uniform(low=-1e-5, high=1e-5, size=(number_of_cells, 3))# to prevent a uniform grid
    pos += del_xyz

pos[:,0] = xx.flatten()
pos[:,1] = yy.flatten()
pos[:,2] = zz.flatten()

"""set up hydrodynamical quantities"""
Mass = np.zeros(number_of_cells) 
Energy = np.zeros(number_of_cells)
Velocity = np.zeros([number_of_cells, 3])
densities = np.zeros(number_of_cells) # This will not go into the final snapshot. But is used to calculate the velocity

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
densities = np.add(densities, density_code)

rad_x = pos[:,0] - 0.5*boxsize
rad_y = pos[:,1] - 0.5*boxsize
rad_z = pos[:,2] - 0.5*boxsize
rad_xy = np.sqrt(rad_x**2 + rad_y**2) + dx/1e6
radii = np.sqrt(rad_x**2 + rad_y**2 + rad_z**2) + dx/1e6

simulation_potentials = total_potential(rad_z, rad_xy, radii)
mid_plane_potentials = total_potential(0, rad_xy, rad_xy)
surface_densities = radial_gas_distribution(rad_xy)

print("disk sound speed", disk_sound_speed)
rs = grid_1d - 0.5*boxsize
z_min = np.min(rad_z)
z_max = np.max(rad_z)
r_bins = np.linspace(dx/100, np.max(rad_xy), 1000) # quads blow up at lower values
quad_vals = np.zeros(r_bins.size)

print("integrating quads") 
for i, rxy in enumerate(r_bins):
    def f(z): # Something to be careful with
        radius_3d = np.sqrt((rxy)**2 + (z)**2) # the radius is also dependant on z
        mid_plane_pot = total_potential(0, rxy, rxy) # mid-plane potential is when z = 0. Note that you can generalize this
        # Note I am getting a "diverges or takes long to converge" warning for the quads if I use total_potential(..), so this has to be seperate
        return np.exp(- ( (total_potential(z, rxy, radius_3d)) - mid_plane_pot ) /disk_sound_speed**2)
    quad_vals[i] = quad(f, z_min, z_max)[0]
print("inserting mass and energy")
for i, rad in enumerate(rad_xy): 
    if i % int(rad_xy.size/10) == 0:
        print("%0.1f" % (i/radii.size * 100), "%") # prints out the percentage every 10%
    if rad_xy[i] <= Rgas*3/2 and np.abs(rad_z[i]) <= 4*zstars_in_UnitLength: # if rad_xy <= are in a cylindrical region of space
        idx = np.abs(r_bins - rad_xy[i]).argmin()
        mid_plane_density = surface_densities[i]/quad_vals[idx]
        density = mid_plane_density*np.exp(-(simulation_potentials[i] - mid_plane_potentials[i])/disk_sound_speed**2) # equation 4 
        densities[i] += density
        Mass[i] += density*pow(dx,3)

        if Mass[i] > mass_code*1.01:
            U = 5/2 * BOLTZMANN_IN_ERG_PER_KELVIN * T_disk_in_Kelvin # The internal energy for a monoatomic ideal gas particle in analytic units
            # Total mass in the cell(in analytic units)/Hydrogen mass in grams(also analytic) = the amount of hydrogen atoms
            U_cell = U*((Mass[i]*UnitMass_in_g/Hydrogen_mass_in_g))/UnitEnergy_in_cgs   # U_cell = U * # of hydrogen molecules in the cell/UnitEnergy_in_cgs
            Energy[i] = U_cell/Mass[i] # Specific internal energy of the cell

# Calculate the pressures and get the gradients
P = densities * (disk_sound_speed**2)/(gamma)
# since we have rad_z == 0 is extremely low(1e-16) but not actually 0 for the milkway, let's swap to np.isclose
Pb, p_edge_r, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], P[np.isclose(rad_z, 0)], bins=301, statistic='mean')
p_gradient_r = np.gradient(Pb, p_edge_r[:-1]) 
rho_r, rho_edge, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], densities[np.isclose(rad_z, 0)], bins=301, statistic='mean')
p_gradient_r = p_gradient_r/rho_r



'''
    - My Personal thoughts as of February 28, 2025: Based on all of my tests, Arepo and CGOLs(or rather, C and Python) calculates everything differently
      What I did below, based on my comparison tests, is what CGOLs did, taking np.gradient(or the equivalent in C) and defining that be
      their acceleration. Arepo, however, seems to calculate the potentials by directly taking the derivative. On paper, this should be leading to the same the
      result. In practice, that's not what's happening.

      2. The NFW is a complicating factor 

'''
# Calculate the halo gradients for r and z. We assume phi gradients are 0 due to axisymmetry
halo_radial, r_edge, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], simulation_potentials[np.isclose(rad_z, 0)], bins=cells_per_dimension, statistic='mean')
potential_gradient_rad = np.gradient(halo_radial, r_edge[:-1]) # dHalo/dr
# halo_z, z_edge, _ = stats.binned_statistic(radii[np.where((rad_x == 0) & (rad_y == 0))], simulation_potentials[np.where((rad_x == 0) & (rad_y == 0))], bins=100, statistic='mean')

acc = np.zeros([number_of_cells,3])
if (STELLAR_POTENTIAL):
    print("Stellar potential enabled, allocating ACs")
    Z = Rstars + np.sqrt(rad_z*rad_z + zstars_in_UnitLength*zstars_in_UnitLength)
    # The Miyamoto-Nagai profile is given in r and z 
    acc[:,0] += -G_code * Mstars * rad_xy / ( pow( (rad_xy*rad_xy + Z*Z) , 3/2) ) * (rad_x/rad_xy)
    acc[:,1] += -G_code * Mstars * rad_xy / ( pow( (rad_xy*rad_xy + Z*Z) , 3/2) ) * (rad_y/rad_xy)
    acc[:,2] += -G_code * Mstars * rad_xy * Z / ( np.sqrt( zstars_in_UnitLength*zstars_in_UnitLength + rad_z*rad_z ) * pow( (rad_xy*rad_xy + Z*Z) , 3/2) )

if (NFW_POTENTIALV2):
    print("NFW potential enabled, allocating ACs")
    acc[:,0] += -G_code * enclosed_mass(radii) * rad_x / (radii * radii * radii)
    acc[:,1] += -G_code * enclosed_mass(radii) * rad_y / (radii * radii * radii)
    acc[:,2] += -G_code * enclosed_mass(radii) * rad_z / (radii * radii * radii)

'''
The acceleration as defined in CGOLS is the momentum equation for fluid flow, where we can reduce to the velocity to purely a azimuthal term: v = r*sqrt(a)
'''

print("inserting acceleration components due to pressure")
for i, r in enumerate(rad_xy): # only in the cylindrical region of our disk
    if i % int(rad_xy.size/10) == 0:
        print("%0.1f" % (i/radii.size * 100), "%") # prints out the percentage every 10.0%

    if(Mass[i] > mass_code*1.01) and 0.99*r < 3/2*Rgas: # The pressure has a massive spike at the edge. 0.99 prevents that spike from happening
        idx_r = np.abs(r_edge[:-1] - r).argmin() # pick the r in dPhi/dr that is closest to us and get index
        idx_pr = np.abs(p_edge_r[:-1] - r).argmin() # same with pressure in r 
        # According the CGOLS paper: "The disk gas is also in _RADIAL EQUILIBRIUM_ with the static potential. Velocities are set by first calculating the tangential acceleration
        # at a given radius due to the gravitational potential, then correcting this acceleration for the the radial pressure gradient.
        ## grad(Phi) = dPhi/dr(r_hat) + 1/r dPhi/dphi(phi_hat) + dPhi/dz(zhat)## 
        ## I am making the assumption that a spinning galactic disk is axisymmetric. 
        ## NOTE: "An axisymmetric flow is defined as one for which the flow variables, ie velocity and pressure, do not vary with the angular coordinate θ."
        ## grad(Phi) = dPhi/dr(r_hat) + dPhi/dz(zhat)

        acc_pressure = p_gradient_r[idx_pr] # -grad_r(Phi) + dP/dr 
        # convert to x and y components
        acc[i,0] += acc_pressure*rad_x[i]/(r) 
        acc[i,1] += acc_pressure*rad_y[i]/(r) 

acc_mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2)
v_circ = np.sqrt(rad_xy*acc_mag)
Velocity[:,0]= -v_circ * rad_y/(rad_xy) # v_x = v_r * cos(theta) - v_theta*sin(theta) -> convert to v_x 
Velocity[:,1] = v_circ * rad_x/(rad_xy) # v_y = v_r * sin(theta) + v_theta*cos(theta) -> convert to v_y 


if (OUTPUT_CSV):
    print("Printing CSVs(This will take a while)")
    # np.savetxt("potential_gradient_stellar.csv", np.column_stack((r_edge[:-1], potential_gradient_rad)), delimiter=",")
    np.savetxt("acc_from_ic_MW.csv", np.column_stack((rad_x, rad_y, rad_z, acc_mag)), delimiter=",")
    if (STELLAR_POTENTIAL) and (NFW_POTENTIAL):
        np.savetxt("potentials_both_mw.csv", np.column_stack((rad_x, rad_y, rad_z, simulation_potentials)), delimiter=",")
    if (STELLAR_POTENTIAL) and (NFW_POTENTIALV2):
        np.savetxt("potentials_both_v2_mw.csv", np.column_stack((rad_x, rad_y, rad_z, simulation_potentials)), delimiter=",")
    if (STELLAR_POTENTIAL):
        np.savetxt("potentials_stellar_mw.csv", np.column_stack((rad_x, rad_y, rad_z, stellar_potential(rad_z, rad_xy))), delimiter=",")
    if (NFW_POTENTIAL):
        np.savetxt("potentials_NFW_mw.csv", np.column_stack((rad_x, rad_y, rad_z, NFW_DM_halo_potential(radii))), delimiter=",")
    if (NFW_POTENTIALV2):
        np.savetxt("potentials_NFWv2_mw.csv", np.column_stack((rad_x, rad_y, rad_z, NFW_DM_halo_potential_v2(radii))), delimiter=",")

if (BACKGROUND_BOX): # Setting up the box, modify the values as needed. Note that there are instances
    bg_midpoint = BG_BOXSIZE/2
    deviation = boxsize/2
    center_l_boundary = bg_midpoint - deviation

    ## Position of first and last cell in the background
    background_cells_per_dimension = 50 # Cells that make up the background grid. 
    dx_bg = BG_BOXSIZE/background_cells_per_dimension
    print("background dx", dx_bg)
    bg_pos_first, bg_pos_last = 0.5*dx_bg, BG_BOXSIZE - 0.5*dx_bg # faces of the cell are placed in between the borders

    p0_bg = Background_T*(dx_bg**3)*(density_0**2)/(UnitDensity_in_cgs)*BOLTZMANN_IN_ERG_PER_KELVIN*UnitMass_in_g/0.63  # Per my derivation, this corresponds to P = 3/2 N^2 kb * T, where N is the number of hydrogen atoms(N^2 is rather strange)

    #### DON'T CHANGE THESE ####
    u_therm_bg = p0_bg/(gamma - 1.0)/density_0 

    # Convert density u_therm_0, and pressure to code units
    density_code = (density_0*Hydrogen_mass_in_g)/(UnitDensity_in_cgs)
    energy_code_bg = (u_therm_bg)/(UnitEnergy_in_cgs) 

    # set up the values for one dimension
    bg_grid_1d = np.linspace(bg_pos_first, bg_pos_last, background_cells_per_dimension)
    background_number_of_cells = pow(background_cells_per_dimension, 3) 
    bg_xx, bg_yy, bg_zz = np.meshgrid(bg_grid_1d, bg_grid_1d, bg_grid_1d)

    # # set up the grid as seen in many Arepo examples
    bg_pos = np.zeros([background_number_of_cells, 3]) 
    bg_pos[:,0] = bg_xx.flatten()
    bg_pos[:,1] = bg_yy.flatten()
    bg_pos[:,2] = bg_zz.flatten()

    bg_mass = np.zeros(background_number_of_cells) 
    bg_energy = np.zeros(background_number_of_cells)

    volume_code_bg = pow(dx_bg,3)
    mass_code_bg = density_code*volume_code_bg
    energy_code_specific_bg = energy_code_bg/mass_code_bg
    bg_mass = np.add(bg_mass, mass_code_bg)
    bg_energy = np.add(bg_energy, energy_code_specific_bg)
    bg_velocity = np.zeros([background_number_of_cells, 3])

    pos += [center_l_boundary, center_l_boundary, center_l_boundary]
    print("min position", np.min(pos))
    print("max position", np.max(pos))

    pos = np.vstack([pos, bg_pos]) 
    Velocity = np.vstack([Velocity, bg_velocity])
    Mass = np.concatenate([Mass, bg_mass])
    Energy = np.concatenate([Energy, bg_energy])

    ### Sorting -> Making sure that all the cells are near each other and in their proper position
    indices = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))

    Velocity = Velocity[indices]
    pos = pos[indices] 
    Mass = Mass[indices]
    Energy = Energy[indices]
    number_of_cells += background_number_of_cells
# import pdb; pdb.set_trace()
"""write *.hdf5 file; minimum number of fields required by Arepo """ 
IC = h5py.File(simulation_directory + 'Disk_M82.hdf5', 'w')
# import pdb; pdb.set_trace()

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

##### DEBUG PLOTS ####
if (OUTPUT_DEBUG_PLOTS):
    print("Printing Debug Plots")
else:
    end = time.time()
    print("Program has finished initial condition files. Time elapsed: ", end - start, "seconds")
    exit()

linear_velocity = np.sqrt(Velocity[:,0]**2 + Velocity[:,1]**2 + Velocity[:,2]**2)
print("maximum velocity magnitude", np.max(linear_velocity))

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(2,2,1)
press, r, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], P[np.isclose(rad_z, 0)], bins=300)
ax1.semilogy(r[:-1], press, color = "darkblue")
ax1.scatter(rad_xy[np.isclose(rad_z, 0)], P[np.isclose(rad_z, 0)], s=5, color = "darkblue")
ax1.set_xlabel("Radial Distance [kpc]")
ax1.set_ylabel(r"Pressure [$\frac{km}{kpc \cdot s^2}$]") 
ax1.set_xlim(0,10)

ax2 = fig.add_subplot(2,2,2)
ax2.plot(p_edge_r[:-1], p_gradient_r , label=r"$\frac{1}{\rho}\frac{dP}{dr}$", color = "darkblue")
ax2.plot(r_edge[:-1],-potential_gradient_rad, label=r"$-\frac{d\Phi}{dr}$", color="red")
ax2.set_xlabel("Radial Distance [kpc]")
ax2.set_ylabel(r"$a_\phi(r,z)$ [$\frac{km}{s^2}$]")
ax2.set_ylim(-22000, 500)
ax2.legend(loc="upper right")
ax2.set_xlim(0,10)

rad_x = pos[:,0] - 0.5*BG_BOXSIZE
rad_y = pos[:,1] - 0.5*BG_BOXSIZE
rad_z = pos[:,2] - 0.5*BG_BOXSIZE
rad_xy = np.sqrt(rad_x**2 + rad_y**2) + dx/1e6
radii = np.sqrt(rad_x**2 + rad_y**2 + rad_z**2) + dx/1e6

ax3 = fig.add_subplot(2,2,3)
v, r_edge_lv, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], linear_velocity[np.isclose(rad_z, 0)], bins=300)
r = 0.5*(r_edge_lv[:-1] + r_edge_lv[1:])
ax3.plot(r, v, label="Magnitude", color = "darkblue")
ax3.scatter(rad_xy[np.isclose(rad_z, 0)], linear_velocity[np.isclose(rad_z, 0)], s=5, color = "darkblue")
v_x, r_edge_x, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], Velocity[:,0][np.isclose(rad_z, 0)], bins=300)
rx = 0.5*(r_edge_x[:-1] + r_edge_x[1:])
ax3.plot(rx, v_x, label=r"$v_x$")
v_y, r_edge_y, _ = stats.binned_statistic(rad_xy[np.isclose(rad_z, 0)], Velocity[:,1][np.isclose(rad_z, 0)], bins=300)
ry = 0.5*(r_edge_y[:-1] + r_edge_y[1:])
ax3.plot(ry, v_y, label=r"$v_y$")
ax3.set_xlabel("Radial Distance [kpc]")
ax3.set_ylabel("Velocity [km/s]") # The linear velocity matches the tangential
ax3.legend() 
ax3.set_ylim(-10, 250)
ax3.set_xlim(0, 10)

ax4 = fig.add_subplot(2,2,4)
ax4.scatter(Velocity[:,1][np.isclose(rad_z, 0)], Velocity[:,1][np.isclose(rad_z, 0)], s=2)
ax4.set_xlabel("$v_x$ [km/s]")
ax4.set_ylabel("$v_y$ [km/s]") 
ax4.set_ylim(-200, 200)
ax4.set_xlim(-200, 200)

plt.savefig(simulation_directory + 'Disk_M82.pdf')
plt.show()

end = time.time()
print("Program has finished initial condition files. Time elapsed: ", end - start, "seconds")
