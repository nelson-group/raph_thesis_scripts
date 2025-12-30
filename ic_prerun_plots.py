"""
    Debugging plots made to check if the initial conditions generated correctly. Plots slices and scatter plots of velocity, density, and temperature
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import glob
from scipy import stats
import matplotlib.colors as colors


#### PHYSICAL CONSTANTS ###
GRAVITIONAL_CONSTANT_IN_CGS = 6.6738e-8
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

#### SIMULATION CONSTANTS - keep it consistent with param.txt ###
UnitVelocity_in_cm_per_s = 1e5 # 1 km/sec 
UnitLength_in_cm = 3.085678e21 # 1 kpc
UnitMass_in_g = 1.989e33 # 1 solar mass
UnitDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**3 # 6.769911178294545e-32 g/cm^3
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.988e43 ergs
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

#### load libraries
import sys    # system specific calls
import numpy as np    ## load numpy
import h5py    ## load h5py; needed to write initial conditions in hdf5 format
from scipy.integrate import quad
from scipy import interpolate
import time
# print("thesis/cc85/create.py: creating ICs in directory" +  simulation_directory)

""" Initial Condition Parameters """
# FilePath = simulation_directory + 'IC300kpc.hdf5'
# Setting up the box, modify the values as needed
boxsize = 10 # Units in kiloparsecs 
cells_per_dimension = 301 # resolution of simulation simulation 
number_of_cells = pow(cells_per_dimension, 3) 
center_boxsize = boxsize/2

# Set up the grid
dx = boxsize/cells_per_dimension # code units

background_boxsize = 100
bg_cells_per_dimension = 50
dx_bg = background_boxsize/bg_cells_per_dimension
midpoint = background_boxsize/2

start_time = time.time()
# The voronoi slice of a certain region(usually the mid-plane) along the z-axis.
def make_voronoi_slice_edge(gas_xyz, gas_values, image_num_pixels, image_y_value, image_xz_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)
    s = image_xz_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    zs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)

    X,Z = np.meshgrid(xs,zs)
    M_coords = np.transpose(np.vstack([X.ravel(), np.full(len(X.ravel()), image_y_value), Z.ravel()]))
    result = np.transpose(interp(M_coords).reshape(len(xs), len(zs)))

    return result, np.array(xs), np.array(zs)

# The voronoi slice of a certain region(usually the mid-plane) along the y-axis.
def make_voronoi_slice_face(gas_xyz, gas_values, image_num_pixels, image_z_value, image_xy_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)
    s = image_xy_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    ys = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s) 

    X,Y = np.meshgrid(xs,ys)
    M_coords = np.transpose(np.vstack([X.ravel(), Y.ravel(), np.full(len(X.ravel()), image_z_value)] ))
    result = np.transpose(interp(M_coords).reshape(len(ys), len(xs)))

    return result, np.array(xs), np.array(ys)

data = {}
filename = "disk_smc_nfw_stellar.hdf5"
with h5py.File(filename,'r') as f:
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
coordinates = data["Coordinates"]
x_coord, y_coord, z_coord = data["Coordinates"][:,0], data["Coordinates"][:,1], data["Coordinates"][:,2]

vel_x = data["Velocities"][:,0]
vel_y = data["Velocities"][:,1] 
vel_z = data["Velocities"][:,2] 
linear_velocities = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy

rad_x = x_coord - 0.5*background_boxsize
rad_y = y_coord - 0.5*background_boxsize
rad_z = z_coord - 0.5*background_boxsize

rad_xy = np.sqrt(rad_x**2+rad_y**2) 
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)

masses = data["Masses"]
densities = np.where(masses == np.max(masses), masses/pow(dx_bg,3), masses/pow(dx,3))
print("background density", np.min(densities*UnitNumberDensity))
deviation = 5.0
histb_l = background_boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = background_boxsize/2  + deviation # boundary of histogram - upper bound
print("boundaries of histogram %f, %f" % (histb_l, histb_h))

face_condition = np.isclose(rad_z, 0) # Note: np.where is slower than boolean indexing.
edge_condition = np.isclose(rad_y, 0)
z_condition = (np.isclose(rad_x, 0) & np.isclose(rad_y, 0))
inside_box = radius <= 10

m, r_edge_m, _ = stats.binned_statistic(rad_xy[face_condition], densities[face_condition]*UnitNumberDensity, statistic='mean', bins=1000)
mz, r_edge_z, _ = stats.binned_statistic(radius[z_condition], densities[z_condition]*UnitNumberDensity , statistic='mean', bins=160)
rm = 0.5 * (r_edge_m[:-1] + r_edge_m[1:])
zm = 0.5 * (r_edge_z[:-1] + r_edge_z[1:])

fig = plt.figure(figsize=(22,13))
fig.set_rasterized(True)

ax1 = fig.add_subplot(2,3,1)
ax1.semilogy(rm, m, label="xy-plane" )
ax1.scatter(rad_xy[face_condition], densities[face_condition]*UnitNumberDensity, s=5)
ax1.semilogy(zm, mz, label="z-plane")
ax1.scatter(radius[z_condition], densities[z_condition]*UnitNumberDensity, s=5)
ax1.set_xlabel("Distance [kpc]")
ax1.set_ylabel("Density [$cm^{-3}$]")
ax1.set_xlim(0,10)
ax1.set_ylim(1e-5, 1e3)
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(2,3,2)
face_d, x_edge_d, y_edge_d = make_voronoi_slice_face(coordinates[face_condition], densities[face_condition], 600, midpoint, center_boxsize)
ax2.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
xz_density = ax2.pcolormesh(x_edge_d, y_edge_d, face_d.T*UnitNumberDensity, cmap='magma',norm=colors.LogNorm(vmin=1e-4,vmax=1e4), shading='auto')
ax2.set_xlabel("X [kpc]")
ax2.set_ylabel("Y [kpc]")
cbar_xy = plt.colorbar(xz_density, ax = ax2, label='N [$cm^{-3}$]')    

# XZ Slice
ax3 = fig.add_subplot(2,3,3)
ax3.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
edge_d, x_edge_d, z_edge_d = make_voronoi_slice_edge(coordinates[edge_condition], densities[edge_condition], 600, midpoint, center_boxsize)
xz_plot = ax3.pcolormesh(x_edge_d, z_edge_d, edge_d.T*UnitNumberDensity, cmap='magma',norm=colors.LogNorm(vmin=1e-4,vmax=1e4), shading='auto')
ax3.set_xlabel("X [kpc]")
ax3.set_ylabel("Z [kpc]")
cbar_xz = plt.colorbar(xz_plot, ax = ax3, label='N [$cm^{-3}$]')    

labels = [1e-4*(10**(x)) for x in range(1,8)]
cbar_xz.set_ticks(labels)
cbar_xz.set_ticklabels([round(np.log10(label)) for label in labels])
cbar_xy.set_ticks(labels)
cbar_xy.set_ticklabels([round(np.log10(label)) for label in labels])

ax4 = fig.add_subplot(2,3,4)
# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

temperature = Temp_S(1, internal_energy)
print(np.min(temperature))
# import pdb; pdb.set_trace()
zstars_in_UnitLength = 0.15 # 0.15 kpc
T, r_edge_T, _ = stats.binned_statistic(rad_xy[face_condition], temperature[face_condition], bins = 1000 )
Tz, z_edge_T, _ = stats.binned_statistic(radius[z_condition], temperature[z_condition], bins=160)
rt = 0.5 * (r_edge_T[:-1] + r_edge_T[1:])
zt = 0.5 * (z_edge_T[:-1] + z_edge_T[1:])

ax4.semilogy(rt, T, label="xy-plane")
ax4.scatter(rad_xy[np.where(np.isclose(rad_z, 0))], temperature[np.where(np.isclose(rad_z, 0))], s=5)
ax4.semilogy(zt, Tz, label="z-plane")
ax4.scatter(radius[z_condition], temperature[z_condition], s=5)

ax4.set_xlim(0,10)
ax4.set_ylim(1e3, 1e8)
ax4.set_xlabel("Distance [kpc]")
ax4.set_ylabel("Temperature [K]")
ax4.text(0.1, 5e7, "Background T = %0.2e K" % np.max(temperature[np.where(np.isclose(rad_z, 0))]))
ax4.text(0.1, 3e7, "Disk T = %0.2e K" % np.min(temperature[np.where(np.isclose(rad_z, 0))]))
ax4.legend(loc='upper right')

#### Velocity Projection in the XY plane
ax5 = fig.add_subplot(2,3,5)
face_v, x_edge_v, y_edge_v = make_voronoi_slice_face(coordinates[face_condition], linear_velocities[face_condition], 600, midpoint, center_boxsize)
ax5.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
print(np.max(linear_velocities))
ax5.set_xlabel("X [kpc]")
ax5.set_ylabel("Y [kpc]")
v_xy_plot = ax5.pcolormesh(x_edge_v, y_edge_v, face_v.T, cmap='viridis', vmin=0, vmax=300, shading='auto')
cbar_vxy = plt.colorbar(v_xy_plot, ax = ax5, label='Velocity Magnitude [km/s]')    

#### Velocity Projection in the XZ plane
ax6 = fig.add_subplot(2,3,6)
ax6.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
edge_v, x_edge_v, z_edge_v = make_voronoi_slice_edge(coordinates[edge_condition], linear_velocities[edge_condition], 600, midpoint, center_boxsize)
ax6.set_xlabel("X [kpc]")
ax6.set_ylabel("Z [kpc]")
v_xz_plot = ax6.pcolormesh(x_edge_v, z_edge_v, edge_v.T, cmap='viridis', vmin=0, vmax=300, shading='auto')
cbar_vxz = plt.colorbar(v_xz_plot, ax = ax6, label='Velocity Magnitude [km/s]')    


def custom_tick_labels(x, pos):
    return f"{x - background_boxsize/2:.0f}"
ax2.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax2.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax3.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax3.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax5.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax5.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax6.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
ax6.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

plt.savefig("Disk_m82_stellar_only_prerun.png", dpi=150, bbox_inches='tight') 
plt.show()
print("generating density distributions across dimensions of the disk")

fig = plt.figure(figsize=(6,5))
fig.set_rasterized(True)

ax1 = fig.add_subplot(1,1,1,)
# np.isclose(rad_y, 0)
# np.isclose(rad_z, 0)
# np.isclose(rad_x, 0)
dx, rx, _ = stats.binned_statistic(rad_x[(np.isclose(rad_y, 0)) & (np.isclose(rad_z, 0))], densities[(np.isclose(rad_y, 0)) & (np.isclose(rad_z, 0))]*UnitNumberDensity, bins = 120)
dy, ry, _ = stats.binned_statistic(rad_y[(np.isclose(rad_x, 0)) & (np.isclose(rad_z, 0))], densities[(np.isclose(rad_x, 0)) & (np.isclose(rad_z, 0))]*UnitNumberDensity, bins = 120)
dz, rz, _ = stats.binned_statistic(rad_z[(np.isclose(rad_x, 0)) & (np.isclose(rad_y, 0))], densities[(np.isclose(rad_x, 0)) & (np.isclose(rad_y, 0))]*UnitNumberDensity, bins = 120)
rx = 0.5 * (rx[:-1] + rx[1:])
ry = 0.5 * (ry[:-1] + ry[1:])
rz = 0.5 * (rz[:-1] + rz[1:])

ax1.semilogy(rx, dx, color="blue", label="x-axis")
ax1.semilogy(ry, dy, color="red", label="y-axis")
ax1.semilogy(rz, dz, color="green", label="z-axis")
ax1.scatter(rad_x[(np.isclose(rad_y, 0)) & (np.isclose(rad_z, 0))], densities[(np.isclose(rad_y, 0)) & (np.isclose(rad_z, 0))]*UnitNumberDensity, s=1, color="blue")
ax1.scatter(rad_z[(np.isclose(rad_x, 0)) & (np.isclose(rad_y, 0))], densities[(np.isclose(rad_x, 0)) & (np.isclose(rad_y, 0))]*UnitNumberDensity, s=1, color="green")

ax1.grid()
ax1.legend(loc="upper right")
ax1.set_xlabel("Distance [kpc]")
ax1.set_ylabel("Density [$cm^{-3}$]")
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(1e-2, 3e2)
plt.savefig("disk_smc_nfw_stellar_distribution.png")
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total elapsed time: {elapsed_time} seconds")