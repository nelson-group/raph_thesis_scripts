'''
    Generates a side-by-side comparison of the edge-on slices of the cooling rates between two snapshots
'''
import h5py
import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import interpolate
import seaborn as sns
import cmasher as cmr
import matplotlib.patches as patches
import matplotlib as mpl
import time 
from matplotlib.ticker import FuncFormatter
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

from matplotlib.colors import SymLogNorm

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS
GRAVITY =  6.6738e-8
s_in_yr = 3.154e+7
#### Configuration Options ####

angle_l = 60
icefire = sns.color_palette("icefire", as_cmap=False)
################################

data = {}
### PARAMETER CONSTANTS AND INITIAL VALUES ###
filename = "./output_PIE_fid/snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 

M_load = parameters["M_load"]
E_load = parameters["E_load"]
sfr = parameters["sfr"]
R = parameters["injection_radius"] # injection radius in kpc
UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178297542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);

alpha_t = 0.90
mu_t = 0.60
beta_t = 0.6
R_inject = 0.3
R_03 = R_inject/0.3
Omega_4pi = (4*np.pi)/(4*np.pi)
zeta = 1 # 0.02/0.02
M_dot = 10
M_stardot = M_dot/10
theta = 60
angle_l = 60
s_in_yr = 3.154e+7
Z_solar = 0.02
boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 300 # general number of bins for the histograms.

inner_boxsize = 10
halfbox = boxsize/2
dx = inner_boxsize/cells_per_dim
devx = dx/1e6
halfbox_inner = inner_boxsize/2 
lower_bound, upper_bound = halfbox - dx*5, halfbox + dx*5

deviation = 15
box_range = inner_boxsize

histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

# The voronoi slice of a certain region(usually the mid-plane) along the z-axis.
def make_voronoi_slice_edge(gas_xyz, gas_values, image_num_pixels, image_y_value, image_xz_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)  # declare an interpolator of coordinates and values associated with the coordinates
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

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log, symnorm):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    if symnorm: edge_mesh.set_norm(SymLogNorm(linthresh=linethreshold, vmin=minimum, vmax=maximum, base=10))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]', fontsize=14)
    ax.set_ylabel('Z [kpc]', fontsize=14)

def calculate_cell_size(volume):
    return 2 * np.cbrt(volume * 3 /(4 * np.pi))

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

linethreshold = 1e-33

### ANALYTIC VALUES
rc = np.linspace(0.001, 10, 1500)
r_10 = rc/10
sfr_10 = sfr/10
advection_time_myr = 1e7 * np.sqrt(M_load/(E_load*mu_t)) * r_10 # * (UnitTime_in_s/s_in_yr) # 1e7 yrs (beta/alpha) * radius/10 kpc # Solar Masses/year
cooling_time_myr = 3e6 * (((E_load*mu_t)**2.20)/(M_load**3.20)) * (R_03/r_10)**0.27 * R_03**2/sfr_10*Omega_4pi
# cooling_time_s = cooling_time_myr*(UnitTime_in_s/s_in_yr)
ratio_an = cooling_time_myr/advection_time_myr
labeling = ["PIE + CIE", "CIE"]
fig = plt.figure(figsize=(11, 5))
fig.set_rasterized(True)    
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
axs = [ax1, ax2]

######### SIMULATION DATA #########
start = time.time()
data = {}
files = ["./output_PIE_fid/snap_100.hdf5", "./output_CIE_fid/snap_100.hdf5"]
for i, file in enumerate(files):
    with h5py.File(file,'r') as f:
        for key in f['PartType0']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
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
    xe = data["ElectronAbundance"]
    metallicities = data["PassiveScalars"]
    print(file)
    cooling_function = data["CoolingRate"] # Lambda/nh^2 
    z_cooling_funtion = data["MetallicCoolingRate"]
    cooling_time = data["CoolingTime"]
    t = header["Time"]
    temperature = Temp_S(1, internal_energy)
    linear_velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    times = t*1000

    volumes = masses/density
    number_density = density*UnitNumberDensity
    density_cgs = density*UnitDensity_in_cgs

    rate_fact = pow(number_density*HYDROGEN_MASS_FRACTION, 2)/density_cgs # cm^-3 * 0.76 

    volumetric_cooling_rate = cooling_function*pow(number_density*HYDROGEN_MASS_FRACTION, 2)
    cooling_rate_erg = cooling_function*rate_fact*masses*UnitMass_in_g # erg cm^3 s^-1 * 

    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) 
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound)
    
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) 
    z_mask = (y_coord >= lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (np.abs(rad_z) <= halfbox_inner) 
    
    advection_time_sim = radius*UnitLength_in_cm/(linear_velocity*UnitVelocity_in_cm_per_s)

    ct_ratio = cooling_time*UnitTime_in_s/advection_time_sim

    tan_theta = rad_x/(rad_z + dx)
    theta = np.arctan(tan_theta)*180/np.pi # this returns -90 to 90 
    angular_region = (np.abs(theta) <= angle_l)  & (np.abs(rad_z) >= 0.3) # & (radius <= 8)# Excludes anything with absolute angles greater than 60 

    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]

    density_rad = density[face_mask]
    density_z = density[z_mask]

    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + dx) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + dx)
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+dx), vel_y - radial_velocity*rad_y/(radial_coord+dx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)
    print(np.min(volumetric_cooling_rate))

    plot_edge(axs[i], coord[edge_mask], volumetric_cooling_rate[edge_mask], n_bins, halfbox, box_range, -1e-30, 1e-24, log=False, symnorm=True)
    cr_mesh = axs[i].collections[0]
    cr_mesh.set_cmap("cmr.iceburn_r")
    cbar = plt.colorbar(cr_mesh, ax = axs[i], pad=0.005)
    cbar.set_label(r'Volumetric Cooling Rate [$\rm erg \,s^{-1} \, cm^{3}$]', fontsize=12)
    background_rect = patches.Rectangle((0, 0.78), width=1, height=0.22, color='black', alpha=0.25, transform=axs[i].transAxes, fill=True)
    axs[i].add_patch(background_rect)
    axs[i].text(0.01, 0.96,"t = %0.1f Myr" % times, transform=axs[i].transAxes, color="white", fontsize=13)
    axs[i].text(0.01, 0.91, labeling[i], transform=axs[i].transAxes, color="white", fontsize=13)

    # ax1.text(0.01, 0.93,r'LMC/M82 Disk - CIE+PIE, High Z', transform=ax1.transAxes, color="white", fontname='serif')
    axs[i].text(0.03, 0.86,r"- $\beta = 0.6, \alpha = 0.9, Z_{disk}= 4 Z_\odot$", transform=axs[i].transAxes, color="white", fontsize=13)
    axs[i].text(0.03, 0.81,r"- $\dot{M}_{SFR} = 20 M_\odot \, yr^{-1}$, $R_{inject}$ = 300 pc", transform=axs[i].transAxes, color="white", fontsize=13)
    
    axs[i].tick_params(axis='both', which='major', labelsize=12)

    # Choose ticks: negative, linear region, and positive
    labels_cooling = [1e-28, -1e-30, -1e-32, 0, 1e-32, 1e-30, 1e-28, 1e-26, 1e-24]
    cbar.set_ticks(labels_cooling)

ax2.set_ylabel("")

plt.tight_layout(w_pad=0.0)
print("generating image for time: ", str(t))
plt.savefig("cooling_edges.pdf", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)