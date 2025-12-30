
# Loading libraries and key coordinates
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import interpolate
from scipy import stats
import seaborn as sns
from matplotlib.ticker import FuncFormatter
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

def make_voronoi_slice_edge(gas_xyz, gas_values, image_num_pixels, image_y_value, image_xz_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)  # declare an interpolator of coordinates and values associated with the coordinates
    s = image_xz_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    zs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)

    X,Z = np.meshgrid(xs,zs)
    M_coords = np.transpose(np.vstack([X.ravel(), np.full(len(X.ravel()), image_y_value), Z.ravel()]))
    result = np.transpose(interp(M_coords).reshape(len(xs), len(zs)))

    return result, np.array(xs), np.array(zs)

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log,):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize)
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Z [kpc]')

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

fig = plt.figure(figsize=(10, 4.0))
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
coordinates = data["Coordinates"]
x_coord = data["Coordinates"][:,0] 
y_coord = data["Coordinates"][:,1]
z_coord = data["Coordinates"][:,2]
density = data["Density"]
masses = data["Masses"]
cooling_function = data["CoolingRate"]
x_e = data["ElectronAbundance"]
internal_energy = data["InternalEnergy"]
vel_x = data["Velocities"][:,0]
vel_y = data["Velocities"][:,1] 
vel_z = data["Velocities"][:,2] 

time = header["Time"]*1000
volume = density/masses 
cell_size = calculate_cell_size(volume)
temperature = Temp_S(x_e, internal_energy)
deviation = center_boxsize/2
rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
number_densities = density*UnitNumberDensity
inner_box = get_interior(rad_x, rad_y, rad_z, center_boxsize)
pressures = data["Pressure"] 
pressure_cgs = pressures*UnitPressure_in_cgs/kb


theta = np.arccos(np.abs(rad_z)/(radius + dx/1e6))*180/np.pi 

angular_region_wind = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 

# Based on eyeballing, the cloud is somewhere between: 
x_cm, x_cmax = 0.8, 1.7
z_cm, z_cmax = 1.3, 2.2

xlimits = (0.0, 3.5)
zlimits = (0.5, 4)

theta_l = 25 
theta_h = 45

y_locate_mask = ((rad_x > x_cm) & (rad_x < x_cmax) & 
                 (rad_z > z_cm) & (rad_z < z_cmax) &  
                #  (theta >= theta_l) & (theta <= theta_h) & 
                 (inner_box) 
                )

radial_velocity = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)
cold_cloud_mask = (y_locate_mask) & (temperature <= T_COLD_MAX)
x_mesh = rad_x[cold_cloud_mask] # so any y that is within range of the cold gas cloud 
y_mesh = rad_y[cold_cloud_mask] # so any y that is within range of the cold gas cloud 
z_mesh = rad_z[cold_cloud_mask] # so any y that is within range of the cold gas cloud 
masses_y = masses[cold_cloud_mask] # weighted by the masses 
x_cloud_center = np.average(x_mesh, weights=masses_y)
y_cloud_center = np.average(y_mesh, weights=masses_y)# so the weighted number that contains where the center of y is.
z_cloud_center = np.average(z_mesh, weights=masses_y)
print(f"cloud center: r_x = %0.2f, r_y = %0.2f, r_z = %0.2f" % (x_cloud_center, y_cloud_center, z_cloud_center))
cloud_variance = np.sqrt(np.average((y_cloud_center - y_mesh)**2, weights=masses_y))
print(cloud_variance) # the cloud center of mass is ~ 1.42 +- 0.58
lower_bound, upper_bound =  midpoint + y_cloud_center - (cloud_variance*2), midpoint + y_cloud_center + (cloud_variance*2)# make sure to include cells that might not be in the center.

cloud_mask =  (
                (rad_y >= y_cloud_center - (cloud_variance*2))   & (rad_y <= y_cloud_center + (cloud_variance*2))  
                & (rad_x >= xlimits[0] ) & (rad_x <= xlimits[1] )
                & (rad_z >= zlimits[0]) & (rad_z <= zlimits[1] )
              )

angular_region = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 

cone_angle_mask = (np.abs(theta) >= 40) & (np.abs(theta) <= 60) 

t_l = 1e5  # ~t_l = np.percentile(temperature[rad_z >= 0.5], 0.15) 
nd_h = 0.0065 # nd_h = np.percentile(number_densities[rad_z >= 0.5], 99.85)  from snapshot 000
nh_c = np.max(number_densities[(angular_region) & (temperature >=  2*T_COLD_MAX)]) # highest density that that's from not from a cloud or a disk
print("cloud density filter", nh_c)
# Define the cloud to be what remains at T_cold_max 
# import pdb; pdb.set_trace()
bg_cells = (number_densities <= nd_h) & (temperature >= t_l) & (np.abs(radial_velocity) <= 40) # Gets rids of as much bg cells  BG cells.
mean_number_density = np.mean(number_densities[(angular_region) & (radius >= R*2) & (number_densities < nh_c) & (radius < 30) & (~bg_cells)]) # to prevent any potential cool clouds& (~bg_cells)]) 
overdensity = (number_densities - mean_number_density)/mean_number_density
bicone_cloud = cloud_mask & inner_box & cone_angle_mask  & (overdensity >= 5) & (temperature <= T_COLD_MAX)
bicone_mixing = cloud_mask & inner_box & cone_angle_mask & (temperature >= T_COLD_MAX) & (temperature <= T_COLD_MAX*10) # & (overdensity >= 5) & (overdensity >= 1)  # & (overdensity > 1) & (overdensity <= 10)
rbins = np.linspace(2.2, 4.0, 12)
print("mean number density", mean_number_density)

T_mixing, r_edge, _ = stats.binned_statistic(radius[bicone_mixing], temperature[bicone_mixing], bins=rbins, statistic="median")
r_faces = 0.5* (r_edge[:-1] + r_edge[1:])
print(T_mixing)

ax1.semilogy(r_faces, T_mixing, color="black", linestyle="dotted", label="Mixing")

T_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], temperature[bicone_cloud], bins=rbins, statistic="median")
print(T_cloud)

ax1.semilogy(r_faces, T_cloud, color="black", label="Cloud")

left_inset, right_inset = 1.8, 4.0
left = (left_inset - 1) / (5 - 1)
width = (right_inset - left_inset) / (5 - 1)
ratio_wind_T_cloud = ax1.inset_axes([0.1, 0.1, 0.86, 0.20], sharex=ax1)
ratio_wind_T_cloud.plot(r_faces, T_cloud/T_mixing, color="black")
ratio_wind_T_cloud.tick_params(axis='x', bottom=True, labelsize=10)
ratio_wind_T_cloud.tick_params(axis='y', bottom=True, labelsize=10)
ratio_wind_T_cloud.set_ylim(0.0, 0.2)
# ratio_wind_T_cloud.set_yticks([0, 1, 2 ])
# ratio_wind_T_cloud.set_xlim(1.8, 3.6)
ratio_wind_T_cloud.set_title(r"$\rm T_{cl}/T_{m}$")


ax1.tick_params(axis='both', which='major', labelsize=12)
x_ticks = [2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
ax1.set_xlabel("Radius [kpc]", fontsize=13)
ax1.set_ylabel(r"Temperature  [K]", fontsize=13)
ax1.legend(loc="upper right", fontsize=12)
ax1.set_xticks(x_ticks)
ax1.set_xlim(2.5, 3.9)
ax1.set_ylim(1e2, 5e6)



ax2 = fig.add_subplot(122)

radial_velocity = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)

inner_rx, inner_ry, inner_rz = rad_x[inner_box], rad_y[inner_box], rad_z[inner_box]
inner_masses = masses[inner_box]
inner_r = np.sqrt(inner_rx**2 + inner_rz**2)
inner_vr = radial_velocity[inner_box]
inner_momentum = inner_masses*inner_vr
cloud_mask_vor =  (
                (rad_y >= y_cloud_center - (dx*12))   & (rad_y <= y_cloud_center + (dx*12)) 
                & (rad_x >= xlimits[0] ) & (rad_x <= xlimits[1] )
                & (rad_z >= zlimits[0]) & (rad_z <= zlimits[1] )
              )
cloud_coords = np.column_stack([rad_x[cloud_mask_vor], rad_z[cloud_mask_vor]])
T_values = temperature[cloud_mask_vor]

interp = interpolate.NearestNDInterpolator(cloud_coords, T_values)

image_num_pixels = 2000
x_grid = np.linspace(xlimits[0], xlimits[1], image_num_pixels)
z_grid = np.linspace(zlimits[0], zlimits[1], image_num_pixels)

GX, GZ = np.meshgrid(x_grid,z_grid)
flattened_grid = np.column_stack([GX.ravel(), GZ.ravel()])

result = np.transpose(interp(flattened_grid).reshape( len(x_grid), len(z_grid) ) )

voronoi_image = result.reshape(GX.shape)
edge_mesh = ax2.pcolormesh(GX, GZ, result.T, shading='auto', cmap="inferno", norm=colors.LogNorm(1e3, 1e6))

cbar = plt.colorbar(edge_mesh, ax =ax2, pad=0.005)
cbar.set_label(r"Temperature [K]", fontsize=13)

ax2.set(xlim=(0, 3.5), ylim=(0.5, 4.0))

ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5])
ax2.set_yticks([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0])

ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)

ax2.set_xlabel('X [kpc]', fontsize=13)
ax2.set_ylabel('Z [kpc]', fontsize=13)

plt.subplots_adjust(wspace=0.2)

plt.savefig("cloud_temperature_PIE_M82.pdf", dpi=300, bbox_inches='tight') # use this to get the mean velocity..

rbins = np.linspace(1, 5, 200)
fig = plt.figure(figsize=(10, 4.0))
fig.set_rasterized(True)
ax1 = fig.add_subplot(121)
p_cloud, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud], bins=rbins, statistic="median")
p_cloud_high, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud],  statistic="max", bins=rbins)
p_cloud_low, r_edge, _ = stats.binned_statistic(radius[bicone_cloud], pressure_cgs[bicone_cloud],  statistic="min", bins=rbins)
r_faces = 0.5* (r_edge[:-1] + r_edge[1:])

P_wind, rwind, _ = stats.binned_statistic(radius[angular_region_wind], pressure_cgs[angular_region_wind], bins=rbins, statistic="median", range=(1, 5))
rw_faces = 0.5* (rwind[:-1] + rwind[1:])

ax1.semilogy(rw_faces, P_wind, color="black", label="Wind")
ax1.semilogy(r_faces, p_cloud, color="royalblue", label="Cloud")


ax1.set_yticks([10**x for x in np.arange(0, 6)])
ax1.legend(loc="upper right", fontsize=11)
ax1.set_ylabel(r"Pressure [$\rm K \, cm^{-3}$]", fontsize=13)
ax1.set_xlabel("Radius [kpc]", fontsize=13)


ratio_wind_press_cloud = ax1.inset_axes([0.08, 0.1, 0.88, 0.20])
ratio_wind_press_cloud.plot(r_faces, p_cloud/P_wind, color="royalblue")
ratio_wind_press_cloud.axhline(1, linestyle="dashed", color="black", linewidth=1)
ratio_wind_press_cloud.tick_params(axis='x', bottom=True, labelsize=9)
ratio_wind_press_cloud.tick_params(axis='y', bottom=True, labelsize=9)
ratio_wind_press_cloud.set_ylim(0.0, 3)
ratio_wind_press_cloud.set_yticks([0, 1, 2, 3])
ratio_wind_press_cloud.set_xlim(1.8, 3.5)
ratio_wind_press_cloud.set_title(r"$\rm P_{cl}/P_w$")
ax1.set_xticks([1,2,3,4,5])
ax1.set_xlim(1,5)
ax1.set_ylim(1, 1e5)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.xaxis.set_tick_params(labelsize=12)

ax2 = fig.add_subplot(122)
xlimits = (0.0, 3.5)
zlimits = (0.5, 4)
P_cloud_mask_vor =  (
                (rad_y >= y_cloud_center - (dx*12))   & (rad_y <= y_cloud_center + (dx*12)) 
                & (rad_x >= xlimits[0] ) & (rad_x <= xlimits[1] )
                & (rad_z >= zlimits[0]) & (rad_z <= zlimits[1] )
              )
P_values = pressure_cgs[P_cloud_mask_vor]
P_cloud_coords = np.column_stack([rad_x[P_cloud_mask_vor], rad_z[P_cloud_mask_vor]])


interp_P = interpolate.NearestNDInterpolator(P_cloud_coords, P_values)
x_grid = np.linspace(xlimits[0], xlimits[1], image_num_pixels)
z_grid = np.linspace(zlimits[0], zlimits[1], image_num_pixels)

GX, GZ = np.meshgrid(x_grid,z_grid)
flattened_grid = np.column_stack([GX.ravel(), GZ.ravel()])
result_P = np.transpose(interp_P(flattened_grid).reshape( len(x_grid), len(z_grid) ) )
voronoi_image = result.reshape(GX.shape)
edge_mesh = ax2.pcolormesh(GX, GZ, result_P.T, shading='auto', cmap="viridis", norm=colors.LogNorm(100, 1e5))
ax2.set_ylabel("Z [kpc]", fontsize=13)
ax2.set_xlabel("X [kpc]", fontsize=13)
cbar = plt.colorbar(edge_mesh, ax =ax2, pad=0.005)
cbar.set_label(r"Pressure [$\rm K cm^{-3}$]", fontsize=13)

ax2.set(xlim=(0, 3.5), ylim=(0.5, 4.0))

ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5])
ax2.set_yticks([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0])

ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)

plt.subplots_adjust(wspace=0.2)
plt.savefig("cloud_P_PIE_M82.pdf", dpi=300, bbox_inches='tight') # use this to get the mean velocity..