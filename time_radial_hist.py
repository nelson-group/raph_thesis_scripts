'''
    This file generates velocity, density, and temperature at 5, 15, 25, and 35 Myrs.
'''

import h5py
import numpy as np    
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from scipy import spatial
from scipy import integrate
from scipy import interpolate
from scipy import optimize
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import time
start_time = time.time()



mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

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

UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 300 # general number of bins for the histograms. Some value <= cells_per dim
R = parameters["injection_radius"] # injection radius in kpc

deviation = 10
histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

### EQUATIONS ###
def make_voronoi_slice(gas_xyz, gas_values, image_num_pixels, image_z_value, image_xy_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values) # declare an interpolator of coordinates and values associated with the coordinates
    # Set up a grid of x adn y values to interpolate over.
    s = image_xy_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s) 
    ys = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)  #

    X,Y = np.meshgrid(xs,ys) # Make a mesh for the values 
    M_coords = np.transpose(np.vstack([X.ravel(), Y.ravel(), np.full(len(X.ravel()), image_z_value)] ))  

    # vectorized interpolatation over every point in the mesh and then reshape the result into the desired grid shape. 
    result = np.transpose(interp(M_coords).reshape(len(ys), len(xs)))

    return result, np.array(xs), np.array(ys)


    # coord = np.transpose(data["Coordinates"])
    # x_coord = data["Coordinates"][:,0] 
    # y_coord = data["Coordinates"][:,1]
    # z_coord = data["Coordinates"][:,2]


# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

## ANALYTIC SOLUTION CALCULATION FOR COMPARISION - CHANGE NUMBERS AS NEEDED ###
sfr = parameters["sfr"]
s_in_yr = 3.154e+7
grams_in_M_sun = 1.989e33
M_load = parameters["M_load"]
M_dot_wind = sfr*M_load # solar masses per 1 year -> get this in grams per second 
E_load = parameters["E_load"]
M_dot = M_dot_wind*grams_in_M_sun/s_in_yr # grams per second 
E_dot_wind = E_load*3e41*sfr # this is in ergs/second 
E_dot = E_dot_wind



######### SIMULATION DATA #########
data = {}
v_rm = np.array([])
M = []
Temperatures = np.array([])


# grab times t = 5,15,25,35 Myr
times = np.array([5, 15, 30])
snaps = times*2 # very roughly, I am taking a snapshot every 0.5 Myrs and since I have 100 snapshots, the snapshot with the closest time will be t*2


fig, axs = plt.subplots(3, 3, figsize=(15, 14))
fig.set_rasterized(True)
pc = [None]*12

x = 0
import time 
start_time = time.time()
def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

for snap in snaps:
    filename = "./snap_%03d.hdf5" % snap
    with h5py.File(filename,'r') as f:
        for key in f['PartType0']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
    coord = data["Coordinates"]
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    density_gradient = data["DensityGradient"] 
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    print(internal_energy)
    masses = data["Masses"] 
    pressures = data["Pressure"] 
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 
    vel = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    E = internal_energy*masses # NOTE: This is the actual internal energy
    t = header["Time"]
    ts = int(round(t*1000))
    ''' Get the radius of the box - will be useful in later plots'''
    rad_x = x_coord - 0.5*boxsize
    rad_y = y_coord - 0.5*boxsize
    rad_z = z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    v_r = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius) 
    temperatures = Temp_S(1, internal_energy)
    '''Get a single slice at the midpoint of the data set in the z-direction'''
    midpoint = boxsize/2
    lower_bound = midpoint - boxsize/(cells_per_dim)
    upper_bound = midpoint + boxsize/(cells_per_dim)
    face_condition = (z_coord >=lower_bound) & (z_coord <= upper_bound)
    n_densities = density[face_condition]
    n_masses = masses[face_condition]
    n_sie = internal_energy[face_condition]
    n_ie = E[face_condition]
    n_x = x_coord[face_condition]
    n_y = y_coord[face_condition]
    n_z = z_coord[face_condition]
    n_vel_r = v_r[face_condition]
    n_temps = temperatures[face_condition]

    
    # 2D VELOCITY HISTOGRAM

    for nn, ax in enumerate(axs.flat):
        if nn == x:
            # print(np.where((z_coord >=lower_bound) & (z_coord <= upper_bound)))
            cmap = plt.cm.magma
            stat, x_edge, y_edge = make_voronoi_slice(coord[face_condition], v_r[face_condition], n_bins, midpoint, boxsize)
            # stat, x_edge, y_edge, bin_n = stats.binned_statistic_2d(n_x, n_y, n_vel_r,  bins = 30, range=[[0,boxsize],[0,boxsize]])
            X, Y = np.meshgrid(x_edge,y_edge) 
            pc[nn] = ax.pcolormesh(X,Y, stat, vmin=0, vmax=1.5e3, cmap=cmap, shading='auto')
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif', fontsize=16)
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
            if x == 1 : plt.setp(ax.get_xticklabels()[0], visible=False)    
            if x < 2: plt.setp(ax.get_xticklabels()[-1], visible=False)    
            # test = interpolate.griddata( method='method')
            # print(test)
        if nn == x + 3: 
            cmap = plt.cm.viridis 
            stat, x_edge, y_edge = make_voronoi_slice(coord[face_condition], density[face_condition], n_bins, midpoint, boxsize)
            #stat, x_edge, y_edge, bin_n = stats.binned_statistic_2d(n_x, n_y, n_densities, bins = n_bins, range=[[0,boxsize],[0,boxsize]])
            X, Y = np.meshgrid(x_edge,y_edge)  
            pc[nn] = ax.pcolormesh(X,Y, stat.T*UnitNumberDensity, cmap=cmap, norm=colors.LogNorm(vmin=1e3*UnitNumberDensity, vmax=1e8*UnitNumberDensity), shading='auto')
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif', fontsize=16)
            if x == 1: plt.setp(ax.get_xticklabels()[0], visible=False)    
            if x < 2: plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        if nn == x + 6:
            cmap = plt.cm.inferno
            stat, x_edge, y_edge = make_voronoi_slice(coord[face_condition], temperatures[face_condition], n_bins, midpoint, boxsize)
            # stat, x_edge, y_edge, bin_n = stats.binned_statistic_2d(n_x, n_y, n_temps, bins = n_bins, range=[[0,boxsize],[0,boxsize]]) # replace with voronoi caculations - see overleaf
            X, Y = np.meshgrid(x_edge,y_edge)
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            pc[nn] = ax.pcolormesh(X,Y, stat.T, cmap=cmap, norm=colors.LogNorm(vmin=1e5, vmax=1e8), shading='auto')
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif', fontsize=16)
            if x == 1: plt.setp(ax.get_xticklabels()[0], visible=False)    
            if x < 2: plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

        if nn in [0,3,6]:
            ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=12)

    x += 1

fig.subplots_adjust(wspace = 0)

cbar1 = fig.colorbar(pc[0], ax=axs[0, :], pad=0.01)
cbar2 = fig.colorbar(pc[4], ax=axs[1, :], pad=0.01)
cbar3 = fig.colorbar(pc[8], ax=axs[2, :], pad=0.01)
cbar1.set_label("Velocity [km/s]",fontsize=16)
cbar2.set_label(r"Density [log($\rm cm^{-3}$)]",fontsize=16)
cbar3.set_label("Temperature [log(K)]",fontsize=16)


labels = [1e3*UnitNumberDensity*(10**(x)) for x in range(1,5)]
cbar2.set_ticks(labels)
cbar2.set_ticklabels([int(np.log10(label)) for label in labels])

labels = [1e5, 1e6, 1e7, 1e8]
cbar3.set_ticks(labels)
cbar3.set_ticklabels([int(np.log10(label)) for label in labels])


fig.text(0.45, 0.07, 'X [kpc]', ha='center', fontsize=16)
fig.text(0.09, 0.5, 'Y [kpc]', va='center', rotation='vertical', fontsize=16)

plt.savefig("hist_evo_fid.pdf", dpi=150, bbox_inches='tight') 
print("Program took seconds", time.time() - start_time, "to run")
plt.show()
