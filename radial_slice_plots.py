'''
    This file generates plots for density, energy, velocity, and temperature as the galactic disk as a function of a radial distance.
    The snapshots here provide a visual representation of the central face of the disk of the galaxy. 

    Set up a sys argv for the run directory
'''
import h5py
import numpy as np    
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
from scipy import interpolate
import matplotlib as mpl
import time 
from matplotlib.ticker import FuncFormatter
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

#### Configuration Options ####
FACE_ON = True
T0_PLOT = False 

################################
# I can just set FACE_ON and do an if else...
if FACE_ON:
    print("FACE_ON enabled.")
else:
    print("FACE_ON disabled. Output will be edge-on")

data = {}
### PARAMETER CONSTANTS ###
filename = "./snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))
    for key in f['PartType0']:
        data[key] = f['PartType0'][key][()]
    header = dict(f['Header'].attrs)
    coord = data["Coordinates"]
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 

UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178297542e-22 barye
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 100 # general number of bins for the histograms.
deviation = 5
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

# Note that you need a lot of other things...
def plot_face(ax, coordinates, value, bins, center, boxsize, minimum, maximum,log):
    stat, x_edge, y_edge = make_voronoi_slice_face(coordinates, value, bins, center, boxsize)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    face_mesh = ax.pcolormesh(x_edge, y_edge, stat.T, shading='auto')
    if (log): face_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: face_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log,):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Z [kpc]')

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

midpoint = boxsize/2
center_boxsize = 10
dx = center_boxsize/cells_per_dim
lower_bound = midpoint - dx
upper_bound = midpoint + dx
mid_center = center_boxsize/2
devx = dx/1e6

###### t = 0.0  values #######
if T0_PLOT:
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    temperature = Temp_S(1, internal_energy)

    face_mask = (z_coord == midpoint) & (radial_coord <= center_boxsize/2*np.sqrt(2)) # max radius is center boxsize/2 *sqrt(3) = 8.66
    edge_mask = (y_coord == midpoint) & (radius <= center_boxsize/2*np.sqrt(3)) # max radius is center boxsize/2 *sqrt(3) = 8.66
    z_mask = (x_coord == midpoint) & (y_coord == midpoint) & (np.abs(rad_z) <= mid_center) 
    
    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]

    density = data["Density"]
    density_rad = density[face_mask]
    density_z = density[z_mask]
    linear_velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + devx)
    tvx = vel_x - radial_velocity*rad_x/(radial_coord+devx)
    tvy = vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)

    if FACE_ON: rho, dist, _ = stats.binned_statistic(r_face, density_rad, bins=n_bins-1)
    else: rho, dist, _ = stats.binned_statistic(r_z, density_z, bins=100)

    rho_init = np.column_stack( (dist[:-1], rho*UnitNumberDensity) )

    if FACE_ON:
        v_r, r_v, _ = stats.binned_statistic(r_face, radial_velocity[face_mask], bins=n_bins-1)
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], bins=n_bins-1 )
        l_v, r_l, _ = stats.binned_statistic(radial_coord[face_mask], linear_velocity[face_mask], bins=n_bins-1)
        tv_init = np.column_stack( (r_t[:-1], v_t) )
    else:
        v_r, r_v, _ = stats.binned_statistic(r_z, radial_velocity_spherical[z_mask], bins=100)
        l_v, r_l, _ = stats.binned_statistic(r_z, linear_velocity[z_mask], bins=100)
    lv_init = np.column_stack( (r_v[:-1], v_r))
    vr_init = np.column_stack( (r_l[:-1], l_v))

    if FACE_ON: T, r, _ = stats.binned_statistic(r_face, temperature[face_mask], bins=n_bins-1 )
    else: T, r, _ = stats.binned_statistic(r_z, temperature[z_mask], bins=100)

    T_init = np.column_stack( (r[:-1], T) )

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
    pressures = data["Pressure"] 
    vel_x = data["Velocities"][:,0]
    vel_y = data["Velocities"][:,1] 
    vel_z = data["Velocities"][:,2] 
    lin_velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    E = internal_energy*masses # NOTE: This is the actual internal energy
    temperature = Temp_S(1, internal_energy)
    t = header["Time"]
    times = t*1000
    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) # max radius is center boxsize/2 *sqrt(3) = 8.66
 
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) & (radial_coord <= center_boxsize/2*np.sqrt(2))
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (radius <= center_boxsize/2*np.sqrt(3))
    z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (np.abs(rad_z) <= mid_center) & (radius <= center_boxsize/2*np.sqrt(3))

    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]

    density_rad = density[face_mask]
    density_z = density[z_mask]

    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + devx)
    tvx = vel_x - radial_velocity*rad_x/(radial_coord+devx)
    tvy = vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)
    rv = radial_velocity[face_mask]
    rv_spherical = radial_velocity_spherical[z_mask]
    ### PLOTS ###
    fig = plt.figure(figsize=(20,12))
    fig.set_rasterized(True)    
    ax1 = fig.add_subplot(2,3,1)
    if FACE_ON: plot_face(ax1, coord[face_mask], density_rad*UnitNumberDensity, n_bins*2, midpoint, center_boxsize, 1e-4, 500, log=True)
    else: plot_edge(ax1, coord[edge_mask], density[edge_mask]*UnitNumberDensity, n_bins*2, midpoint, center_boxsize, 1e-4, 500, log=True)
    density_mesh = ax1.collections[0]
    density_mesh.set_cmap("magma")
    cbar = plt.colorbar(density_mesh, ax = ax1, label='Density [log($cm^{-3}$)]')
    labels = [1e-4*(10**(x)) for x in range(1,7)]
    cbar.set_ticks(labels)
    cbar.set_ticklabels([round(np.log10(label)) for label in (labels)])
    ax1.text(0.01, 0.97,"t = %0.3f Myr" % times, transform=ax1.transAxes, color="white", fontname='serif')
    ### Insert text here ###
    if FACE_ON: ax1.text(0.01, 0.93,'Disk M82, Moving Mesh - Face-on', transform=ax1.transAxes, color="white", fontname='serif')
    else: ax1.text(0.01, 0.93,'Medium Disk - Outflows - Edge-on', transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.89,r'- $\beta=0.25, \alpha=0.25$', transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.86,'- $R_{inject}=1.0$', transform=ax1.transAxes, color="white", fontname='serif')
    ax1.text(0.03, 0.83,r'- $Volume Refinement$ ', transform=ax1.transAxes, color="white", fontname='serif')

    # 2D VELOCITY CENTER VORONOI SLICE 
    ax2 = fig.add_subplot(2,3,2)
    if FACE_ON: plot_face(ax2, coord[face_mask], tan_velocity[face_mask], n_bins*2, midpoint, center_boxsize, 0, 1000, log=False)
    else: plot_edge(ax2, coord[edge_mask], radial_velocity_spherical[edge_mask], n_bins*2, midpoint, center_boxsize, -10, 1000, log=False)
    velocity_mesh = ax2.collections[0]
    if FACE_ON: velocity_mesh.set_cmap("viridis")
    else: velocity_mesh.set_cmap("Spectral")
    cbar = plt.colorbar(velocity_mesh, ax = ax2, label='Radial Velocity [km/s]') 

    # 2D TEMPERATURE CENTER VORONOI SLICE 
    ax3 = fig.add_subplot(2,3,3)
    if FACE_ON: plot_face(ax3, coord[face_mask], temperature[face_mask], n_bins*2, midpoint, center_boxsize, 1e3, 1e7, log=True)
    else: plot_edge(ax3, coord[edge_mask], temperature[edge_mask], n_bins*2, midpoint, center_boxsize, 1e3, 1e7, log=True)
    T_mesh = ax3.collections[0]
    T_mesh.set_cmap("plasma")
    cbar3 = plt.colorbar(T_mesh, ax = ax3, label='Temperature [log(K)]')
    labels = [1e3*(10**(x)) for x in range(0,4)]
    cbar3.set_ticks(labels)
    cbar3.set_ticklabels([int(np.log10(label)) for label in labels]) 

    # DENSITY RADIAL PROFILE
    ax4= fig.add_subplot(2,3,4)
    if FACE_ON:
        ax4.set_xlabel("Radial Distance [kpc]")
        density_stat, r_edge_d, _ = stats.binned_statistic(r_face, density_rad, bins = 200, range=[1e-3, 15])
        sorted_indices = np.argsort(r_face)
        rcs = r_face[sorted_indices]
        rrhos = density_rad[sorted_indices]
        ax4.semilogy(rcs, density_rad*UnitNumberDensity, label="Non-Profiled t = $%0.1f$ Myr" % times, color="lightgray")
        ax4.semilogy(r_edge_d[:-1], density_stat*UnitNumberDensity, label="Profile t = $%0.1f$ Myr" % times, color='midnightblue') 
        if T0_PLOT: ax4.semilogy(rho_init[:,0], rho_init[:,1], label="Profile t = 0.0 Myr", linestyle="dashed", color='midnightblue') 
    else:   
        ax4.set_xlabel("z [kpc]")
        rho_z, z_rho, _ = stats.binned_statistic(r_z, density[z_mask], bins=80)
        rz = rad_z[np.argsort(radius)]
        zrho = density[z_mask]
        sorted_indices = np.argsort(r_z)
        zcs= r_z[sorted_indices]
        zrhos = zrho[sorted_indices]
        ax4.semilogy(zcs, zrhos*UnitNumberDensity, label="Non-Profiled t = $%0.1f$ Myr" % times, color="lightgray")
        ax4.semilogy(z_rho[:-1], rho_z*UnitNumberDensity, color='midnightblue', label = r"t = $%0.1f$ Myr" % times) 
        if T0_PLOT: ax4.semilogy(rho_init[:,0], rho_init[:,1], label="Profile t = 0.0 Myr", linestyle="dashed", color='midnightblue') 
    ax4.set_ylabel("Density [log($cm^{-3}$)]")
    ax4.set_ylim(1e-5,1000)
    if FACE_ON: ax4.set_xlim(0, 7)
    else: ax4.set_xlim(0, 5)
    ax4.legend(loc='upper right')

    # VELOCITY RADIAL PROFILE 
    ax5 = fig.add_subplot(2,3,5)
    ax5.set_xlim(0, 5)
    ax5.set_ylabel("Velocity [km/s]")
    if FACE_ON:
        ax5.set_xlabel("Radial Distance [kpc]")
        ax5.set_ylim(-50, 270) # For a relaxed disk that is settled into equilibrium, the radial velocity should be around 0.
        
        v_r, r_v, _ = stats.binned_statistic(r_face, rv, bins=n_bins-1)
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], bins=n_bins-1 )
        l_v, r_l, _ = stats.binned_statistic(r_face, lin_velocity[face_mask], bins=n_bins-1)
        rvs = rv[np.argsort(rc)]

        ax5.plot(rcs, rvs, label="Radial(Non-Profiled) t = $%0.1f$ Myr" % times, color="lightgray")
        ax5.plot(r_v[:-1], v_r, label="Radial(profiled) t = $%0.1f$ Myr" % times, color='midnightblue') 
        ax5.plot(r_t[:-1], v_t, label="Circular t = $%0.1f$ Myr" % times, color='red') #
        ax5.plot(r_l[:-1], l_v, label="Linear t = $%0.1f$ Myr" % times, color="black") 
        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='Radial t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if T0_PLOT: ax5.plot(lv_init[:,0], lv_init[:,1], label="Linear t = 0.0 Myr", linestyle="dashed", color="black")
        if T0_PLOT: ax5.plot(tv_init[:,0], tv_init[:,1], label="Circular t = 0.0 Myr" % times, color='black', linestyle="dashed")
    else:
        ax5.set_ylim(-100, 1000)
        ax5.set_xlabel("z [kpc]")
        v_r, r_v, _ = stats.binned_statistic(r_z, radial_velocity_spherical[z_mask], bins=80)
        l_v, r_l, _ = stats.binned_statistic(r_z, lin_velocity[z_mask], bins=80)
        zv = radial_velocity_spherical[z_mask]
        zvs = zv[np.argsort(zc)]
        ax5.plot(zcs, zvs, label="Radial(Non-Profiled) t = $%0.1f$ Myr" % times, color="lightgray")
        ax5.plot(r_v[:-1], v_r, label='Radial t = $%0.1f$ Myr' % times, color='midnightblue') 
        ax5.plot(r_l[:-1], l_v, label="Linear t = $%0.1f$ Myr" % times, color="black") 
        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='Radial t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if T0_PLOT: ax5.plot(lv_init[:,0], lv_init[:,1], label="Linear t = 0.0 Myr", linestyle="dashed", color="black")
    if FACE_ON: ax5.set_xlim(0, 7)
    else: ax5.set_xlim(0, 5)
    ax5.legend(loc='upper right')

    # TEMPERATURE RADIAL PROFILE
    ax6 = fig.add_subplot(2,3,6) 
    if FACE_ON:
        ax6.set_xlabel("Radial Distance [kpc]")
        T, r, _ = stats.binned_statistic(r_face, temperature[face_mask], bins=n_bins-1 )
        ax6.semilogy(r[:-1], T, label="t = $%0.1f$ Myr" % times, color='midnightblue')
        if T0_PLOT: ax6.semilogy(T_init[:,0], T_init[:,1], label='t = 0.0 Myr', color='midnightblue', linestyle="dashed")
    else:
        ax6.set_xlabel("z [kpc]")
        T, r_T, _ = stats.binned_statistic(r_z, temperature[z_mask], bins=80)
        ax6.semilogy(r_T[:-1], T, color='midnightblue', label='$%0.1f$ Myr' % times)
        if T0_PLOT: ax6.semilogy(T_init[:,0], T_init[:,1], label='t = 0.0 Myr', color='midnightblue', linestyle="dashed")
    if FACE_ON: ax6.set_xlim(0, 7)
    else: ax6.set_xlim(0, 5)
    ax6.set_ylim(1e4,1e8)
    ax6.legend(loc="upper right")
    ax6.set_ylabel("Temperature [K]")

    # SAVING THE IMAGES FOR TIMESTEP t 
    if FACE_ON: img_name = "face_t" + "%0.5f" % t
    else: img_name = "edge_t" + "%0.5f" % t
    print("generating image for time: ", str(t))
    simulation_directory = str(sys.argv[1]) 

    plt.savefig(simulation_directory + img_name + ".png", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)