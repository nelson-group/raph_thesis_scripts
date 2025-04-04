'''
    This file generates plots for velocity, density, and temperature at 5, 15, 25, and 35 Myrs.
'''

import h5py
import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import interpolate
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
n_bins = 500 # general number of bins for the histograms. Some value <= cells_per dim
R = parameters["injection_radius"] # injection radius in kpc

deviation = 10
histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

### EQUATIONS ###
def make_voronoi_slice(gas_xyz, gas_values, image_num_pixels, image_z_value, image_xy_minmax): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)
    s = image_xy_minmax/image_num_pixels
    xs = np.arange(0, image_xy_minmax+s, s)
    ys = np.arange(0, image_xy_minmax+s, s) 

    X,Y = np.meshgrid(xs,ys)
    M_coords = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    result = np.zeros((len(xs), len(ys)))

    for c in M_coords:
        n_v = interp([c[0], c[1], image_z_value])
        result[np.where(xs == c[0]), np.where(ys == c[1])] = n_v
    return result, xs, ys

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

fig = plt.figure(figsize=(22, 12))
fig.set_rasterized(True)


######### SIMULATION DATA #########
data = {}
v_rm = np.array([])
M = []
Temperatures = np.array([])

# grab times t = 5,15,25,35 Myr
times = np.array([5, 15, 25, 35])
snaps = times*2 # very roughly, I am taking a snapshot every 0.5 Myrs and since I have 100 snapshots, the snapshot with the closest time will be t*2


fig, axs = plt.subplots(3, 4, figsize=(20, 14))
fig.set_rasterized(True)
pc = [None]*12

x = 0
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

    def custom_tick_labels(x, pos):
        return f"{x - boxsize/2:.0f}"

    for nn, ax in enumerate(axs.flat):
        if nn == x:
            cmap = plt.cm.magma
            stat, x_edge, y_edge = make_voronoi_slice(coord, v_r, n_bins, midpoint, boxsize)
            X, Y = np.meshgrid(x_edge,y_edge) 
            pc[nn] = ax.pcolormesh(X,Y, stat, vmin=0, vmax=1.5e3, cmap=cmap, shading='auto')
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif')
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
            if x != 0:
                plt.setp(ax.get_xticklabels()[0], visible=False)    
            plt.setp(ax.get_xticklabels()[-1], visible=False)    
        if nn == x + 4: 
            cmap = plt.cm.viridis 
            stat, x_edge, y_edge = make_voronoi_slice(coord, density, n_bins, midpoint, boxsize)
            X, Y = np.meshgrid(x_edge,y_edge)  
            pc[nn] = ax.pcolormesh(X,Y, stat.T*UnitNumberDensity, cmap=cmap, norm=colors.LogNorm(vmin=1e3*UnitNumberDensity, vmax=1e8*UnitNumberDensity), shading='auto')
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif')
            if x != 0:
                plt.setp(ax.get_xticklabels()[0], visible=False)    
            plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        if nn == x + 8:
            cmap = plt.cm.inferno
            stat, x_edge, y_edge = make_voronoi_slice(coord, temperatures, n_bins, midpoint, boxsize)
            X, Y = np.meshgrid(x_edge,y_edge)
            ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
            pc[nn] = ax.pcolormesh(X,Y, stat.T, cmap=cmap, norm=colors.LogNorm(vmin=1e5, vmax=1e8), shading='auto')
            ax.text(0.02, 0.93,'t =' + str(ts) + " Myr" , transform=ax.transAxes, color="white", fontname='serif')   
            if x != 0:
                plt.setp(ax.get_xticklabels()[0], visible=False)    
            plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

        if nn in [0,4,8]:
            ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        else:
            ax.set_yticklabels([])
    x += 1

fig.subplots_adjust(wspace = 0)

cbar1 = fig.colorbar(pc[0], ax=axs[0, :], label="Velocity [km/s]", pad=0.01)
cbar2 = fig.colorbar(pc[4], ax=axs[1, :], label="Density [log($cm^{-3}$)]",  pad=0.01)
cbar3 = fig.colorbar(pc[8], ax=axs[2, :], label="Temperature [log(K)]",  pad=0.01)

labels = [1e3*UnitNumberDensity*(10**(x)) for x in range(1,5)]
cbar2.set_ticks(labels)
cbar2.set_ticklabels([int(np.log10(label)) for label in labels])

labels = [1e5, 1e6, 1e7, 1e8]
cbar3.set_ticks(labels)
cbar3.set_ticklabels([int(np.log10(label)) for label in labels])

fig.text(0.45, 0.08, 'X [kpc]', ha='center')
fig.text(0.1, 0.5, 'Y [kpc]', va='center', rotation='vertical')

plt.savefig("hist_evo_15Myr_voronoi_500.pdf", dpi=150, bbox_inches='tight') 
plt.show()