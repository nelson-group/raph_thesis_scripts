'''

Takes every ten snapshots in the simulation and generates plots for tangential/circular velocity, density, and temperature of the center x-y plane/face.

'''

import h5py
import numpy as np    
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
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
v_rm = np.array([])
M = []

legends = []
colors = []
linestyles = []

files = glob.glob('./snap_*.hdf5')

snaps = [0, 10, 20, 30, 40, 50, 60, 70, 80 , 90, 100]

avg_vel = np.zeros(shape=(len(snaps), n_bins))
avg_density = np.zeros(shape=(len(snaps), n_bins))
avg_pressure = np.zeros(shape=(len(snaps), n_bins))
avg_temperature = np.zeros(shape=(len(snaps), n_bins))

for i, snap in enumerate(snaps):
    filename = "./snap_%03d.hdf5" % snap
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
    temp = Temp_S(1, internal_energy)

    times = np.append(times, t*1000)
    print(times)

    zstars_in_UnitLength = 0.15
    
    midpoint = boxsize/2
    condition = np.where((z_coord == midpoint))

        # VELOCITY DATA    
    vel_stat, r_edge_v, bin_n = stats.binned_statistic(radial_coord[condition], tan_velocity[condition], bins = n_bins, range=[0, boxsize])
    avg_vel[i] = vel_stat

    # DENSITY DATA  
    density_stat, r_edge_d, bin_n = stats.binned_statistic(radial_coord[condition], density[condition], bins = n_bins, range=[0, boxsize])
    avg_density[i] = density_stat*UnitDensity_in_cgs/PROTON_MASS_GRAMS

    # PRESSURE DATA
    pressure_stat, r_edge_p, bin_n = stats.binned_statistic(radial_coord[condition], pressures[condition], bins = n_bins, range=[0, boxsize])
    avg_pressure[i] = pressure_stat*UnitPressure_in_cgs/kb

    # TEMPERATURE 
    temperature_stat, r_edge_t, bin_n = stats.binned_statistic(radial_coord[condition], temp[condition], bins = n_bins, range=[0, boxsize])
    avg_temperature[i] = temperature_stat

color_map = plt.get_cmap('jet')
colors = color_map(np.linspace(0, 1, len(avg_vel)))

fig = plt.figure(figsize=(15,4))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

for i in range(0, len(avg_vel)):
    ax1.plot(r_edge_v[:-1], avg_vel[i], label="t = %i Myrs" % times[i], color = colors[i])
ax1.set(xlim=(0, 15), ylim=(0, 500)) 
ax1.set_ylabel("Tangential Velocity [km/s]")
ax1.set_xlabel("Distance [kpc]")
ax1.yaxis.set_label_coords(-0.131, 0.5)

for i in range(0, len(avg_density)):
    ax2.semilogy(r_edge_d[:-1], avg_density[i], label="t = %i Myrs" % times[i], color = colors[i])
ax2.set(xlim=(0, 15), ylim=(1e-4,100))
ax2.set_ylabel("N [$cm^{-3}$]")
ax2.set_xlabel("Distance [kpc]")
ax2.minorticks_off()
ax2.yaxis.set_label_coords(-0.12, 0.5)

for i in range(0, len(avg_temperature)):
    ax3.semilogy(r_edge_t[:-1], avg_temperature[i], label="t = %i Myrs" % times[i], color = colors[i])
ax3.set(xlim=(0, 15),ylim=(1e3,1e9))
ax3.set_ylabel("Temperature [K]")
ax3.set_xlabel("Distance [kpc]")
ax3.minorticks_off()
ax3.yaxis.set_label_coords(-0.12, 0.5)

ax1.legend(fontsize='x-small', loc='upper right', ncol=3)   
ax2.legend(fontsize='x-small', loc='upper right', ncol=3) 
ax3.legend(fontsize='x-small', loc='upper right', ncol=3)

plt.savefig("time_evolution_disk.pdf", dpi=150, bbox_inches='tight') 
plt.show()