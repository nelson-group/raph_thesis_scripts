import matplotlib.colors as colors
# Loading libraries and key coordinates
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import cmasher as cmr

plt.rcParams['legend.title_fontsize'] = 'large'
################################################
##### PHYSICAL CONSTANTS #####
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS or erg/K
seconds_in_myrs = 3.15576e+13
seconds_in_yrs = 3.154e+7
sb_constant_cgs = 5.670374419e-5 # ergs/cm^2/K^4/s
T_COLD_MAX = 3e4
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS
z_solar = 0.02

##### UNITS #####
UnitVelocity_in_cm_per_s = 1e5 # 10 km/sec  
UnitLength_in_cm = 3.085678e21 # 1 kpc
UnitMass_in_g  = 1.989e33 # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS
UnitEnergyDensity = UnitEnergy_in_cgs/pow(UnitLength_in_cm, 3)

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)



###### KEY SIMULATION PARAMETERS ######
boxsize = 100
midpoint = boxsize/2 
center_boxsize = 10
cells_per_dim = 301 
deviation = 10
histb_l = boxsize/2 - deviation  # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

angle_l = 60
dx = center_boxsize/300
eps = dx/1e6
Z_solar = 0.02


dx = center_boxsize/cells_per_dim
halfbox_inner = center_boxsize/2 
lower_bound, upper_bound = midpoint - dx*5, midpoint + dx*5
######### SIMULATION DATA #########
data = {}
labeling = []

time_labels = r"t = 50 Myr"
text_label = r"M82 Disk, $Z = 4 Z_\odot$, $\dot{M}_{sfr} = 20 M_\odot \, \, yr^{-1}$"

plt.style.use("seaborn-v0_8-bright")
fig = plt.figure(figsize=(7,6))
fig.set_rasterized(True)    
ax1 = fig.add_subplot(111)

data_init = {}
filename ="./snap_000.hdf5"
with h5py.File(filename,'r') as f:
    for key in f['PartType0']:
        data_init[key] = f['PartType0'][key][()]
x_coord_0 = data_init["Coordinates"][:,0] 
y_coord_0 = data_init["Coordinates"][:,1]
z_coord_0 = data_init["Coordinates"][:,2]
abundance_0 = data_init["ElectronAbundance"]
number_density_0 = data_init["Density"]*UnitNumberDensity
internal_energy_0 = data_init["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
temperature_0 = Temp_S(abundance_0, internal_energy_0)
rad_x0, rad_y0, rad_z0 = x_coord_0 - 0.5*boxsize, y_coord_0 - 0.5*boxsize, z_coord_0 - 0.5*boxsize
nd_h = np.percentile(number_density_0[rad_z0 >= 2], 99.85)
t_l = np.percentile(temperature_0[rad_z0 >= 2], 0.15)


file = "./snap_100.hdf5"
with h5py.File(file,'r') as f:
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    parameters = dict(f['Parameters'].attrs)
R = parameters["injection_radius"]
E_load = parameters["E_load"]
M_laod = parameters["M_load"]

coord = data["Coordinates"]
x_coord = data["Coordinates"][:,0] 
y_coord = data["Coordinates"][:,1]
z_coord = data["Coordinates"][:,2]
density = data["Density"]
internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
masses = data["Masses"] 
vel_x = data["Velocities"][:,0]
vel_y = data["Velocities"][:,1] 
vel_z = data["Velocities"][:,2] 
number_density = density*UnitNumberDensity
abundance = data["ElectronAbundance"]
temperature = Temp_S(abundance, internal_energy)
t = header["Time"]
times = t*1000
''' Get the radial distance of the box'''
rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
radial_velocity = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps) # Units: km/s

edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) # & (radius <= center_boxsize/2*np.sqrt(3))
theta = np.arccos(np.abs(rad_z)/(radius + eps))*180/np.pi 


bg_cells = (number_density <= nd_h) & (temperature >= t_l) & (np.abs(radial_velocity) <= 40) # Gets rids of most BG cells.

angular_region = (np.abs(theta) <= angle_l) & ((np.abs(rad_z) >= R)) & (radius <= 40) # Excludes anything with absolute angles greater than 60 
angular_bg_mask = (angular_region) & (~bg_cells)   


log10_nd = np.log10(number_density)
log10_temperature = np.log10(temperature)

rho_bins = np.linspace(-5, 1, 300)
T_bins = np.linspace(3, 8, 300)

td, nd_edge, _ = stats.binned_statistic(log10_nd[angular_region], log10_temperature[angular_region], bins=rho_bins, statistic="median")

stat_phase_nd, xc_nd, zc_nd, bin_nd = stats.binned_statistic_2d(log10_nd[angular_region],
                                                        log10_temperature[angular_region], 
                                                        masses[angular_region],
                                                        bins=[rho_bins, T_bins], 
                                                        statistic="sum")

phase_plot_hist_rho = ax1.pcolormesh(xc_nd, zc_nd, stat_phase_nd.T, cmap="cmr.cosmic", shading='auto', norm=colors.LogNorm(vmin=0.01, vmax=1e7))

labeling = r"$\beta = %.1f, \alpha = %.1f$" % (parameters["M_load"], parameters["E_load"])
ax1.text(-4.9, 7.6, time_labels, fontsize=14)
ax1.text(-4.9, 7.35, r"$\dot{M}_{\rm SFR} = 20$ Myr, $\rm Z = 4 Z_\odot$", fontsize=14)
ax1.text(-4.9, 7.1, labeling, fontsize=14)

labels_density = [x for x in range(-5, 3, 2)]
labels_temperature = [x+0.5 for x in range(3, 8)]
labels_P = [x for x in range(1,8, 2)]
labels_erho = [x for x in range(-27, -16, 2)]
ax1.set_xticks(labels_density)
ax1.set_yticks(labels_temperature)

ax1.set_ylabel(r"Temperature [$\rm log_{10}{K}$]", fontsize=14)
ax1.set_xlabel(r"Density [$\rm log_{10}(cm^{-3})$]", fontsize=14)
ax1.set(xlim=(-5,1), ylim=(3.5, 7.8))


cbar = plt.colorbar(phase_plot_hist_rho, ax=ax1, pad=0.01)
cbar.set_label(r'Masses  [$\rm M_\odot$]', fontsize=14)

cbar.ax.tick_params(labelsize=12)

ax1.tick_params(axis='both', which='major', labelsize=12)

plt.savefig("pd_n_T_fid_params.pdf", dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
