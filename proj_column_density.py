
# Loading libraries and key coordinates
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
import cmasher as cmr

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
z_solar = 0.02

plt.style.use("seaborn-v0_8-bright")

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


###### KEY SIMULATION PARAMETERS ######
boxsize = 100
midpoint = boxsize/2 
center_boxsize = 10
center_boxsize_large = 15 
cells_per_dim = 301 
cells_per_dim_large = 451 # for the MW plots
deviation = 5
# deviation_large = 7.5
histb_l = boxsize/2 - deviation  # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

angle_l = 60
dx = center_boxsize/300
eps = dx/1e6
Z_solar = 0.02

file = "./snap_100.hdf5" 
title_text = r"M82/LMC - PIE Cooling" # r"Milky Way - Continious Starburst", r"SMC - Continious Starburst"]
data = {}
center_boxsize = 20 # size of the inner box in kpc
#### FUNCTIONS ####
def calculate_cell_size(volume):
    return 2 * np.cbrt(volume * 3 /(4 * np.pi))

def get_interior(rx, ry, rz, center_boxsize):
    return (np.abs(rx) <= center_boxsize) & (np.abs(ry) <=  center_boxsize) & (np.abs(rz) <=  center_boxsize) 

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)


fig = plt.figure(figsize=(10, 4))
fig.set_rasterized(True)
ax1 = fig.add_subplot(121)

deviation_cd = 4
histb_lcd = boxsize/2 - deviation_cd  
histb_hcd = boxsize/2  + deviation_cd   # boundary of histogram - upper bound

masses_array = []
column_densities = []
rads_cd = []
surface_densities = []
pixel_areas = []
with h5py.File(file,'r') as f:
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    parameters = dict(f['Parameters'].attrs)
M_load = parameters["M_load"]
E_load = parameters["E_load"]
boxsize = parameters["BoxSize"]
R = parameters["injection_radius"]
sfr = parameters["sfr"]
x_coord = data["Coordinates"][:,0] 
y_coord = data["Coordinates"][:,1]
z_coord = data["Coordinates"][:,2]
density = data["Density"]
masses = data["Masses"]
cooling_function = data["CoolingRate"]
x_e = data["ElectronAbundance"]
internal_energy = data["InternalEnergy"]
time = header["Time"]*1000
volume = density/masses 
cell_size = calculate_cell_size(volume)
temperature = Temp_S(x_e, internal_energy)
deviation = center_boxsize/2
rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 

inner_box = get_interior(rad_x, rad_y, rad_z, center_boxsize)

# print(np.min(temperature[(rad_z > 1.0*R) & (radius <= 2.77)]))
theta = np.arccos(np.abs(rad_z)/(radius + dx/1e6))*180/np.pi 
angular_region = (np.abs(theta) <= 60) # & (radius <= 20) # & (np.abs(rad_z) >= R) #  & (np.abs(rad_z) >= 0.3) # Excludes anything with absolute angles greater than 60 and the injection radius


inner_rx_cold, inner_ry_cold, inner_rz_cold = rad_x[(inner_box) & (temperature <= 3e4)], rad_y[(inner_box) & (temperature <= 3e4)], rad_z[(inner_box) & (temperature <= 3e4)]
inner_masses_cold = masses[(inner_box) & (temperature <= 3e4)]
inner_r_cold = np.sqrt(inner_rx_cold**2 + inner_rz_cold**2)
sum_masses_c, x_edge_c, z_edge_c, binnumber =  stats.binned_statistic_2d(inner_rx_cold, inner_rz_cold, inner_masses_cold, statistic="sum", bins=1200)



pixel_width_c = (x_edge_c[1] - x_edge_c[0]) 
pixel_height_c = (z_edge_c[1] - z_edge_c[0]) 
pixel_area_c = pixel_width_c * pixel_height_c

surface_density_c =  sum_masses_c/pixel_area_c # M_odot/kpc^2  


nh_column_density_c = surface_density_c * (HYDROGEN_MASS_FRACTION/PROTON_MASS_GRAMS)*UnitMass_in_g/pow(UnitLength_in_cm, 2)
nh_c = nh_column_density_c[(x_edge_c[:-1] >= 0.8) & (x_edge_c[1:] <= 1.7), :][:, (z_edge_c[:-1] >= 1.3) & (z_edge_c[1:] <= 2.2)]
x_edge_c_sub = x_edge_c[(x_edge_c >= 0.8) & (x_edge_c <= 1.7)]
z_edge_c_sub = z_edge_c[(z_edge_c >= 1.3) & (z_edge_c <= 2.2)]


# to calculate the radial profiles of the column density of the cold cloud specificall between 0.8 to 1.7kpc in x and 1.3 to 2.2 kpc in z
x_c = 0.5 * (x_edge_c[1:] + x_edge_c[:-1])
z_c = 0.5 * (z_edge_c[1:] + z_edge_c[:-1])


xlimits = (0.5, 2.6)
zlimits = (0.8, 3.6)

expected_x = (x_c >= 0.8) &  (x_c <= 2.0)  # & (nh_column_density > 0)
expected_z = (z_c >= 1.2) &  (z_c <= 3.0) # & (nh_column_density > 0)
# print(x_c[expected_x])
# print(z_c[expected_z])
condition = (nh_column_density_c > 0.0)
nh0_cd = nh_column_density_c
nh0_cd[~condition] = np.nan
# create a subgrid of values: 
nh0_cd = nh0_cd[np.ix_(expected_x, expected_z)]
# print(nh0_cd)
Xc, Zc = np.meshgrid(x_c[expected_x], z_c[expected_z])
radial_distance = np.sqrt(Xc**2 + Zc**2)
print(np.max(radial_distance))
filter = np.isfinite(nh0_cd.T.ravel()) # & (nh0_cd.T.ravel() > 0)#  &  # nh_sub.T matches Xc/Zc layout
# print(np.unique(nh0_cd.T.ravel()[~filter]))
tan_theta = Xc.ravel()/Zc.ravel()  # tan_theta = rad_x/(rad_z + dx/1e6)
theta = np.arctan(tan_theta)*180/np.pi # this returns -90 to 90 

angle_region_cloud = (theta[filter] <= 42) & (theta[filter] >= 30) 
#  (theta < 90-30) & (theta > 90-42) 

rd_cloud = radial_distance.ravel()[filter][angle_region_cloud]
nh_cloud = nh0_cd.T.ravel()[filter][angle_region_cloud]
print(np.max(nh_cloud))

from scipy.ndimage import median_filter
from scipy.ndimage import percentile_filter

rbins = np.linspace(np.min(radial_distance), np.max(radial_distance), 20)
cd_ma, r, _ = stats.binned_statistic(rd_cloud, nh_cloud, statistic="max", bins=rbins)
cd_n, r, _ = stats.binned_statistic(rd_cloud, nh_cloud, statistic="median", bins=rbins)

ax1.semilogy(r[:-1], cd_n, color="red", label=r"Cloud Edge")
ax1.semilogy(r[:-1], cd_ma, color='black', label=r"Cloud Center")

ax1.tick_params(axis='both', which='major', labelsize=11)




ax1.set_xlim(1.5, 3.5)
ax1.set_ylim(1e19, 1e22)

ax1.set_xlabel("Radial Distance [kpc]", fontsize=12)
ax1.set_ylabel(r"Column Density [$\rm cm^{-2}$]", fontsize=12)
ax1.legend(loc="upper right", fontsize=12)




ax2 = fig.add_subplot(122)

inner_rx, inner_ry, inner_rz = rad_x[inner_box], rad_y[inner_box], rad_z[inner_box]
inner_masses = masses[inner_box]
inner_r = np.sqrt(inner_rx**2 + inner_rz**2)
sum_masses, x_edge, z_edge, binnumber =  stats.binned_statistic_2d(inner_rx, inner_rz, inner_masses, statistic="sum", bins=1200)

pixel_width = (x_edge[1] - x_edge[0]) 
pixel_height = (z_edge[1] - z_edge[0]) 
pixel_area = pixel_width * pixel_height  # get the actual area in kpc^2

number_density = density*UnitNumberDensity

ax2.set(xlim=(0.5, 2.6), ylim=(1.0, 3.6))


surface_density =  sum_masses/pixel_area # M_odot/kpc^2  

nh_column_density = surface_density * (HYDROGEN_MASS_FRACTION/PROTON_MASS_GRAMS)*UnitMass_in_g/pow(UnitLength_in_cm, 2)
nh_column_density = nh_column_density.T
print(np.shape(nh_column_density))

x_face = 0.5 * (x_edge[:-1] + x_edge[1:])
Z_face = 0.5 * (z_edge[:-1] + z_edge[1:])

X, Z = np.meshgrid(x_face, Z_face)
tan_theta = X.ravel()/Z.ravel()  # tan_theta = rad_x/(rad_z + dx/1e6)
theta = np.arctan(tan_theta)*180/np.pi # this returns -90 to 90 

theta = np.reshape(theta, np.shape(X))
print(np.shape(theta))

log10_nhcd = np.log10(nh_column_density)
log10_nhcd = np.where(np.isfinite(log10_nhcd), log10_nhcd, 0)# since log10(0) is not finite.


cd_plot = ax2.pcolormesh(x_edge, z_edge, log10_nhcd, vmin=20, vmax=21.5, shading='auto')

cbar = plt.colorbar(cd_plot, ax =ax2, pad=0.005)
cbar.set_label(r"Column Density [$\rm log_{10}(cm^{-2})$]", fontsize=12)
cd_plot.set_cmap('cmr.arctic')
ax2.text(0.01, 0.94,"t = %0.1f Myr" % time, transform=ax2.transAxes, color="white", fontsize="large")
ax2.text(0.01, 0.89,title_text, transform=ax2.transAxes, color="white", fontsize="large")
ax2.text(0.01, 0.84,r"$\alpha = %0.2f, \beta = %0.2f$" % (E_load, M_load), transform=ax2.transAxes, color="white", fontsize="large")
injection_radius_pc = R*1000
ax2.text(0.01, 0.79,r"$\dot{M}_{\rm SFR} = %0.1f M_\odot \, yr^{-1}$" % sfr, transform=ax2.transAxes, color="white", fontsize="large")
ax2.text(0.01, 0.74,r"$R_{\rm injection} = %0.0f$ pc" % injection_radius_pc, transform=ax2.transAxes, color="white", fontsize="large")

ax2.set_xlabel('X [kpc]', fontsize=12)
ax2.set_ylabel('Z [kpc]', fontsize=12)

ax2.set(xlim=xlimits, ylim=zlimits)

labels = [20, 20.5, 21, 21.5]
cbar.set_ticks(labels)
ax2.set(xlim=(0.5, 2.6), ylim=(1, 3.6))
ax2.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])

cbar.ax.tick_params(labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)

plt.subplots_adjust(wspace=0.2)

Tl = pow(10, 5.49)
Th = pow(10, 5.51)
print(np.median(cooling_function[(temperature > Tl) & (temperature < Th)]))

plt.savefig("projection_column_density_PIE_M82_updated.pdf", dpi=300, bbox_inches='tight')


