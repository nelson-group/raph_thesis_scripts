'''
    Generates slices for the cooling rate, metallicity, and electron abundances.
    Set up a sys argv for the run directory
'''
import h5py
import sys
import glob
import time 
import numpy as np  
import seaborn as sns
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy import interpolate

mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

# linthresh = 1e-24  # Set this to a value appropriate for your data
# face_mesh = ax.pcolormesh(
#     x_edge, y_edge, stat.T, shading='auto',
  


### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

#### Configuration Options ####
FACE_ON = False
simulation_directory = str(sys.argv[1]) 

icefire = sns.color_palette("icefire", as_cmap=False)

################################
if FACE_ON: print("FACE_ON enabled.")
else: print("FACE_ON disabled. Output will be edge-on")

data = {}
### PARAMETER CONSTANTS AND INITIAL VALUES ###
filename = "./snap_000.hdf5" 
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
R = parameters["injection_radius"] # injection radius in kpc
UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178297542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 150 # general number of bins for the histograms.
deviation = 5
histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

inner_boxsize = 10 
halfbox = boxsize/2
dx = inner_boxsize/cells_per_dim
devx = dx/1e6
halfbox_inner = inner_boxsize/2 
lower_bound, upper_bound = halfbox - dx*5, halfbox + dx*5
linethresh = 1e-25

### EQUATIONS ###
# Analytic function - based off of equation 8 of Nguyen et. al, 2022
# Solution inside the injection radius
##### Taken from Nguyen et. al 2023
# def sol_in(M, r):
#     T1 = ((2 + M**2 * (gamma - 1) )/(gamma + 1))**(-(1 + gamma) / (2*(-5*gamma - 1)))
#     T2 = ((1 + 3*gamma*M**2)/(1 + 3*gamma))**((-3*gamma - 1)/(5*gamma - 1))
#     return M*T1*T2 - r/R
def sol_in(M, r):
    T1 = ((3*gamma + 1/M**2)/(1+3*gamma))**(-(3*gamma+1)/(5*gamma+1))
    T2 = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma+1)/(2*(5*gamma+1)))
    return T1*T2 - r/R

# Solution outside the injection radius
##### Taken from Chevalier and Clegg 85
def sol_out(M, r):
    T = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma + 1)/(2*(gamma - 1)))
    result = M**(2/(gamma - 1))*T - (r/R)**2
    return result

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

# Note that you need a lot of other things...
def plot_face(ax, coordinates, value, bins, center, boxsize, minimum, maximum,log, symnorm):
    stat, x_edge, y_edge = make_voronoi_slice_face(coordinates, value, bins, center, boxsize)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    face_mesh = ax.pcolormesh(x_edge, y_edge, stat.T, shading='auto')
    if (log): face_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    if symnorm: face_mesh.set_norm(SymLogNorm(linthresh=linethresh, vmin=minimum, vmax=maximum, base=10))
    else: face_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log, symnorm):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    if symnorm: edge_mesh.set_norm(SymLogNorm(linthresh=linethresh, vmin=minimum, vmax=maximum, base=10))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Z [kpc]')

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"


######### SIMULATION DATA #########
start = time.time()
data = {}
files = glob.glob('./snap_*.hdf5')
for i in np.arange(0,  len(files)): # select the snapshot range to go through
    filename = "./snap_%03d.hdf5" % i
    with h5py.File(filename,'r') as f:
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
    temperature = Temp_S(1, internal_energy)
    t = header["Time"]
    times = t*1000

    volumes = masses/density
    number_density = density*UnitNumberDensity
    metallicities = data["Metallicity"]
    # NOTE: The online documentation is lying, this is not Lambda in cgs units

    cooling_rate = data["CoolingRate"] # For now..newer simulations return the negaive by default. 
    z_cooling_rate = data["MetallicCoolingRate"]

    xe = data["ElectronAbundance"]


    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) 
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) & (radial_coord <= inner_boxsize/2*np.sqrt(2))
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
    z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (np.abs(rad_z) <= halfbox_inner) 

    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]

    density_rad = density[face_mask]
    density_z = density[z_mask]

    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + devx)
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+devx), vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)


    ### PLOTS ###
    fig = plt.figure(figsize=(20,12))
    fig.set_rasterized(True)    
    ax1 = fig.add_subplot(2,3,1)

    if FACE_ON: plot_face(ax1, coord[face_mask], cooling_rate[face_mask], n_bins*2, halfbox, inner_boxsize, -1e-23, 1e-23, log=False, symnorm=True)
    else: plot_edge(ax1, coord[edge_mask], cooling_rate[edge_mask], n_bins*2, halfbox, inner_boxsize, -1e-23, 1e-23, log=False, symnorm=True)
    cr_mesh = ax1.collections[0]
    cr_mesh.set_cmap("cmr.iceburn_r")
    cbar = plt.colorbar(cr_mesh, ax = ax1, label='$\Lambda_{net}$ [$(erg \,\, cm^3 s^{-1})$]')
    background_rect = patches.Rectangle((0, 0.8), width=1, height=0.2, color='black', alpha=0.25, transform=ax1.transAxes, fill=True)
    ax1.add_patch(background_rect)
    ax1.text(0.01, 0.97,"t = %0.3f Myr" % times, transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.01, 0.93,'M82/LMC - Disk Z Test', transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.89,r"- $Z_{bg}= 0.00, \alpha = 0.6$", transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.86,r"- $Z_{disk}= 0.02, \beta = 0.6$", transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.825,r"- $\dot{M}_{SFR} = 10 M_\odot \, yr^{-1}$, $R_{inject}$ = 300 pc", transform=ax1.transAxes, color="white", fontname='serif')
    
    # Choose ticks: negative, linear region, and positive
    labels = [-1e-23,-1e-24, -1e-25, 0, 1e-25, 1e-24, 1e-23]
    cbar.set_ticks(labels)

    # 2D VELOCITY CENTER VORONOI SLICE 
    ax2 = fig.add_subplot(2,3,2)
    if FACE_ON: plot_face(ax2, coord[face_mask], metallicities[face_mask], n_bins*2, halfbox, inner_boxsize, 0, 1.0, log=False,  symnorm=False)
    else: plot_edge(ax2, coord[edge_mask], metallicities[edge_mask], n_bins*2, halfbox, inner_boxsize, 0.0, 1.0, log=False,  symnorm=False)
    Z_mesh = ax2.collections[0]
    Z_mesh.set_cmap("cmr.ember")
    # if FACE_ON: cbar = plt.colorbar(Z_mesh, ax = ax2, label='Metallicity') 
    cbar = plt.colorbar(Z_mesh, ax = ax2, label='Metallicity') 
    labels = [0.00, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(labels)

    # 2D TEMPERATURE CENTER VORONOI SLICE 
    ax3 = fig.add_subplot(2,3,3)
    if FACE_ON: plot_face(ax3, coord[face_mask], xe[face_mask], n_bins*2, halfbox, inner_boxsize, 0, 0.3, log=False, symnorm=False)
    else: plot_edge(ax3, coord[edge_mask], xe[edge_mask], n_bins*2, halfbox, inner_boxsize, 0, 0.3, log=False, symnorm=False)
    xe_mesh = ax3.collections[0]
    xe_mesh.set_cmap("cmr.amethyst")
    cbar3 = plt.colorbar(xe_mesh, ax = ax3, label=r'$\rm n_e/n_H$')
    labels = [0.0, 0.01, 0.2, 0.3]
    cbar3.set_ticks(labels)

    ax4= fig.add_subplot(2,3,4)
    if FACE_ON:
        sSc, sSr,  _ = stats.binned_statistic(radial_coord , cooling_rate, statistic='mean', bins=200)
        col_sScnh = np.where(sSc > 0, "blue", "crimson")
        sSc[sSc < 0] *= -1 # the cooling rate should be posible always 
        NC_points = np.array([sSr[:-1], np.log10(sSc)]).T
        crS_nh = np.array([NC_points[:-1], NC_points[1:]]).transpose(1, 0, 2)
        crS_nhlc = LineCollection(crS_nh, colors=col_sScnh[:-1], linewidth=1.5, label=r"$\Lambda_{net}$", linestyle="solid")
        ax4.set_ylim(-29,-20)
        ax4.add_collection(crS_nhlc)
        ax4.set_xlabel("Radial Distance [kpc]")
    else:
        sSc, sSr,  _ = stats.binned_statistic(radius , cooling_rate, statistic='mean', bins=200)
        col_sScnh = np.where(sSc > 0, "blue", "crimson")
        sSc[sSc < 0] *= -1 # the cooling rate should be posible always 
        NC_points = np.array([sSr[:-1], np.log10(sSc)]).T
        crS_nh = np.array([NC_points[:-1], NC_points[1:]]).transpose(1, 0, 2)
        crS_nhlc = LineCollection(crS_nh, colors=col_sScnh[:-1], linewidth=1.5, label=r"$\Lambda_{net}$", linestyle="solid")
        ax4.add_collection(crS_nhlc)
        ax4.set_xlabel("Radius [kpc]")
        ax4.set_ylim(-26,-20)
    ax4.set(xlim=(0,5), ylabel="$\Lambda_{net}$ [$log_{10}(erg \,\, cm^3 s^{-1})$]")
    ax4.legend(loc='upper right')

    ax5= fig.add_subplot(2,3,5)
    if FACE_ON:
        sSc, sSr,  _ = stats.binned_statistic(radius , metallicities, statistic='mean', bins=200)
        ax5.plot(sSr[:-1], sSc, label="Metallicity", color='midnightblue')
        ax5.set_xlabel("Radial Distance [kpc]")
    else:
        sSc, sSr,  _ = stats.binned_statistic(radius , metallicities, statistic='mean', bins=200)
        ax5.plot(sSr[:-1], sSc, label="Metallicity", color='midnightblue')
        ax5.set_xlabel("Radius [kpc]")
    ax5.set(xlim=(0,5), ylim=(0.0,5), ylabel="Metallicity")
    ax5.legend(loc='upper right')

    ax6 = fig.add_subplot(2,3,6)
    if FACE_ON:
        sSc, sSr,  _ = stats.binned_statistic(radius[face_mask] , xe[face_mask], statistic='mean', bins=200)
        ax6.plot(sSr[:-1], sSc, label=r'$x_e$', color='midnightblue')
        ax6.set(xlim=(0,5), ylim=(0,1.5), ylabel=r'$\rm n_e/n_H$', xlabel="Radial Distance [kpc]")
        ax6.legend(loc='upper right')
    else:
        sSc, sSr,  _ = stats.binned_statistic(radius , xe, statistic='mean', bins=200)
        ax6.plot(sSr[:-1], sSc, label=r'$x_e$', color='midnightblue')
        ax6.set(xlim=(0,5), ylim=(0,1.5), ylabel=r'$\rm n_e/n_H$', xlabel="Radius [kpc]")
        ax6.legend(loc='upper right')

    # # SAVING THE IMAGES FOR TIMESTEP t 
    if FACE_ON: img_name = "cooling_face_t" + "%0.5f" % t
    else: img_name = "cooling_edge_t" + "%0.5f" % t
    # img_name = "cooling_t" + "%0.5f" % t
    print("generating image for time: ", str(t))
    plt.savefig(simulation_directory + img_name + ".png", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)