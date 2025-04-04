'''
Generates the profiles of velocity, density, and pressure along the central x-y plane and z axis. Used to verify initial conditions
'''

import h5py
import numpy as np    
import matplotlib.pyplot as plt
from scipy import stats

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

### PARAMETER CONSTANTS ###
filename = "./snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))

UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 300 # general number of bins for the histograms. Some value <= cells_per dim

def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

######### SIMULATION DATA #########
data = {}
times = np.array([])

filename = "./snap_000.hdf5"
with h5py.File(filename,'r') as f:
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    parameters = dict(f['Parameters'].attrs)

boxsize = parameters["BoxSize"] # boxsize in kpc
dx = boxsize/cells_per_dim
coord = np.transpose(data["Coordinates"])
x_coord = data["Coordinates"][:,0] 
y_coord = data["Coordinates"][:,1]
z_coord = data["Coordinates"][:,2]
density = data["Density"]
density_gradient = data["DensityGradient"] 
internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
masses = data["Masses"] 
pressures = data["Pressure"] 

''' Get the radius of the box'''
rad_x = x_coord - 0.5*boxsize
rad_y = y_coord - 0.5*boxsize
rad_z = z_coord - 0.5*boxsize
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
radial_coord = np.sqrt(rad_x**2 + rad_y**2)

vel_x = data["Velocities"][:,0]
vel_y = data["Velocities"][:,1] 
vel_z = data["Velocities"][:,2] 

vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
v_r = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + dx/1000) # radial velocity.

tvx = vel_x - v_r*rad_x/(radial_coord+dx/1000)
tvy = vel_y - v_r*rad_y/(radial_coord+dx/1000)
tvz = vel_z - v_r*rad_y/(radial_coord+dx/1000)

tan_velocity = np.sqrt(tvx**2 + tvy**2)

E = internal_energy*masses # NOTE: This is the actual internal energy
t = header["Time"]
img_name = "init_profile_t" + "%0.5f" % t

temp = Temp_S(1, internal_energy)

times = np.append(times, t*1000)

midpoint = boxsize/2 # middle of the box 

r_condition = np.where((z_coord == midpoint))
z_condition = np.where((x_coord == midpoint) & ((y_coord == midpoint)))
radial_cond = radial_coord[r_condition]
tan_vr = tan_velocity[r_condition]
densities_r = density[r_condition]
temp_r = temp[r_condition] 

z_cond = radius[z_condition] + 1e-2
tan_vz = tan_velocity[z_condition]
densities_z = density[z_condition]
temp_z = temp[z_condition]

# VELOCITY DATA    
vel_stat, r_edge_v, bin_n = stats.binned_statistic(radial_cond, tan_vr, bins = n_bins, range=[0, 15])

# DENSITY DATA  
density_stat, r_edge_d, bin_n = stats.binned_statistic(radial_cond, densities_r, bins = n_bins, range=[0, 15])

# TEMPERATURE 
temperature_stat, r_edge_t, bin_n = stats.binned_statistic(radial_cond, temp_r, bins = n_bins, range=[0, 15])

fig = plt.figure(figsize=(15,4))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(r_edge_v[:-1], vel_stat, label="xy-plane")
ax1.plot(np.sort(z_cond), tan_vz[np.argsort(z_cond)], label="z-plane") 
ax1.set(xlim=(1e-2, 8), ylim=(-10, 250)) 
ax1.set_ylabel("Circular Velocity [km/s]")
ax1.set_xlabel("Distance [kpc]")
ax1.yaxis.set_label_coords(-0.131, 0.5)

ax2.loglog(r_edge_d[:-1], density_stat*UnitDensity_in_cgs/PROTON_MASS_GRAMS, label="xy-plane")
ax2.loglog(np.sort(z_cond), densities_z[np.argsort(z_cond)]*UnitDensity_in_cgs/PROTON_MASS_GRAMS, label="z-plane")
ax2.set(xlim=(1e-2, 10), ylim=(1e-4,1000))
ax2.set_ylabel("N [$cm^{-3}$]")
ax2.set_xlabel("Distance [kpc]")
ax2.minorticks_off()
ax2.yaxis.set_label_coords(-0.12, 0.5)

ax3.loglog(r_edge_t[:-1], temperature_stat, label="xy-plane")
ax3.loglog(np.sort(z_cond), temp_z[np.argsort(z_cond)] , label="z-plane")
ax3.set(xlim=(1e-2, 10),ylim=(1e3,1e9))
ax3.set_ylabel("Temperature [K]")
ax3.set_xlabel("Distance [kpc]")
ax3.minorticks_off()
ax3.yaxis.set_label_coords(-0.12, 0.5)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')

plt.savefig( img_name + ".pdf", dpi=150, bbox_inches='tight') 
plt.show()