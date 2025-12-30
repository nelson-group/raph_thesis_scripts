'''
    This file generates plots for density, energy, velocity, and temperature as the galactic disk as a function of a radial distance.
    The snapshots here provide a visual representation of the central edge of the disk of the galaxy. 

    Set up a sys argv for the run directory
'''
import h5py
import time 
import sys
import numpy as np    
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Arc
from scipy import stats
from scipy import interpolate
from scipy import optimize
from matplotlib.ticker import FuncFormatter
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS
z_solar = 0.02

#### Configuration Options ####
FACE_ON = True
T0_PLOT = True 
CC85_PLOTS = False
EXTENDED = False
COOLING = True
simulation_directory = str(sys.argv[1]) 

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
    abundance = data["ElectronAbundance"]
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
InitDiskMetallicity = parameters["InitDiskMetallicity"]
boxsize = parameters["BoxSize"] # boxsize in kpc
inner_boxsize = 10
angle_l = 60
halfbox = boxsize/2
dx = inner_boxsize/cells_per_dim
eps = dx/1e4
halfbox_inner = inner_boxsize/2 
lower_bound, upper_bound = halfbox - dx*6, halfbox + eps*6

if EXTENDED: 
    n_bins = 450
    deviation = 50
    box_range = 100
    z_binning = 300
    upper_x = 50
else: 
    deviation = 5
    box_range = inner_boxsize
    z_binning = 100
    upper_x = 5
    n_bins = 300

histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

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

# print(Temp_S(1.2, 20))
# import pdb;pdb.set_trace()
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
    face_mesh = ax.pcolormesh(x_edge, y_edge, stat.T, shading='auto')
    if (log): face_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: face_mesh.set_clim(minimum, maximum)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')

def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, log,):
    stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center, boxsize)
    edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
    if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
    else: edge_mesh.set_clim(minimum, maximum)
    ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
    ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax.set_xlabel('X [kpc]', fontsize=13)
    ax.set_ylabel('Z [kpc]', fontsize=13)

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

## ANALYTIC SOLUTION CALCULATION FOR COMPARISION - CHANGE NUMBERS AS NEEDED ###
if (CC85_PLOTS):
    M_load = parameters["M_load"]
    E_load = parameters["E_load"]
    R = parameters["injection_radius"]
    sfr = parameters["sfr"]

    r_an = np.linspace(0.001, boxsize, 1500)
    r_in = r_an[np.where(r_an <= R)]
    r_out = r_an[np.where(r_an > R)]

    s_in_yr = 3.154e+7
    grams_in_M_sun = 1.989e33
    M_dot_wind = sfr*M_load # solar masses per 1 year -> get this in grams per second 
    M_dot_cm = (M_dot_wind*UnitMass_in_g)/s_in_yr # grams/second
    E_dot_wind = E_load*3e41*sfr # this is in ergs/second 

    M_dot_code = M_dot_wind/(UnitMass_in_g/grams_in_M_sun)*(UnitTime_in_s/s_in_yr)
    E_dot_code = E_dot_wind/UnitEnergy_in_cgs*UnitTime_in_s

    M1 = optimize.fsolve(sol_in, x0=np.full(len(r_in), 0.001), args=(r_in))
    M2 = optimize.fsolve(sol_out, x0=np.full(len(r_out), 100), args=(r_out))
    M = np.concatenate([M1, M2])

    v_an = (M*np.sqrt(E_dot_code/M_dot_code)*(((gamma - 1)*M**2 + 2)/(2*(gamma - 1)))**(-0.5)) # this is in code units
    v_in = v_an[np.where(r_an <= R)]
    v_out = v_an[np.where(r_an > R)]
    cs = np.sqrt( (E_dot_code/M_dot_code)*(((gamma - 1)*M**2 + 2)/(2*(gamma - 1)))**(-1))
    cs_cm = cs*(UnitVelocity_in_cm_per_s) 

    rho_in = M_dot_code/(4*np.pi*v_in)*(r_in/R**3)*UnitDensity_in_cgs
    rho_out = M_dot_code/(4*np.pi*v_out)*1/r_out**2*UnitDensity_in_cgs

    rho_an = np.concatenate([rho_in, rho_out])
    rho_n = np.concatenate([rho_in, rho_out])/PROTON_MASS_GRAMS # rho/(proton mass)

    pressure_an = ((rho_an*cs_cm**2)/gamma)/kb # -> (g/cm^3* cm^2/s^2) -> p/kb 

    # P/kb = rho/(mean molecular weight * proton mass) * T = P/kb = rho/(proton mass) * 1/mean molecular weight * T
    ## T = pressure_an/(rho_n)* mean molecular weight
    temp_an = pressure_an/(rho_n)*(mean_molecular_weight(1)/PROTON_MASS_GRAMS) # keep the mean molecular mass the same. 

###### t = 0.0  values #######
if T0_PLOT:
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    temperature = Temp_S(1, internal_energy)
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) & (radial_coord <= inner_boxsize/2*np.sqrt(2))
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
    if EXTENDED: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)
    else: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]
    density_z = density[z_mask]

    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + eps) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)
    tvx = vel_x - radial_velocity*rad_x/(radial_coord + eps)
    tvy = vel_y - radial_velocity*rad_y/(radial_coord + eps)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)
    rv_z = radial_velocity_spherical[z_mask]
    temp_z = temperature[z_mask]
    # initial density profile
    if FACE_ON: rho, dist, _ = stats.binned_statistic(r_face, density[face_mask], bins=n_bins)
    else: rho, dist, _ = stats.binned_statistic(r_z, density_z, bins=n_bins)
    rho_init = np.column_stack( (dist[:-1], rho*UnitNumberDensity) ) # initial density as a 2d array

    # initial velocity profiles
    if FACE_ON:
        v_r, r_v, _ = stats.binned_statistic(r_face, radial_velocity[face_mask], bins=n_bins) # radial velocity
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], bins=n_bins) # tangential or circular velocity 
        tv_init = np.column_stack( (r_t[:-1], v_t) ) # tangential velocity as a 2d array
    else:
        v_r, r_v, _ = stats.binned_statistic(r_z, rv_z, bins=n_bins) # radial velocity 
    vr_init = np.column_stack((r_v[:-1], v_r)) # radial velocity as a 2d array

    # initial temperature profiles
    if FACE_ON: T, r, _ = stats.binned_statistic(r_face, temperature[face_mask], bins=n_bins)
    else: T, r, _ = stats.binned_statistic(r_z, temp_z, bins=n_bins)
    T_init = np.column_stack( (r[:-1], T) )

######### SIMULATION DATA #########
start = time.time()
data = {}
for i in np.arange(0, 101, 1): # select the snapshot range to go through
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
    sfr = parameters["sfr"]
    
    # import pdb; pdb.set_trace()
    if COOLING: abundance = data["ElectronAbundance"]
    else: abundance = 1
    volume = masses/density
    number_density = density*UnitNumberDensity

    temperature = Temp_S(abundance, internal_energy)
    t = header["Time"]
    times = t*1000
    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2 + rad_y**2+ rad_z**2)
    # print(radius[masses == np.max(masses)])

    # t_l = np.percentile(temperature[rad_z >= 0.5],) 
    # nd_h = np.percentile(number_density[rad_z >= 0.5], 99.85)
    print(np.min(temperature))
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) 


    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) # & (radial_coord <= inner_boxsize/2*np.sqrt(2))
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) # & (radius <= inner_boxsize/2*np.sqrt(3))
    if EXTENDED: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)
    else: z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound) & (radius <= inner_boxsize/2*np.sqrt(3))
    r_face = radial_coord[face_mask] 
    r_z = radius[z_mask]

    temp_z = temperature[z_mask]
    density_rad = density[face_mask]
    density_z = density[z_mask]
    volume = masses/density
    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + eps) 
    radial_velocity_spherical = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/(radius + eps)
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+dx), vel_y - radial_velocity*rad_y/(radial_coord+dx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)
    rv_z = radial_velocity_spherical[z_mask]

    theta = np.arccos(np.abs(rad_z)/(radius + eps))*180/np.pi 
    angular_region = (np.abs(theta) <= 60) # Excludes anything with absolute angles greater than 60 
    ### PLOTS ###
    fig = plt.figure(figsize=(15,9)) # 20/12 = 15/9
    fig.set_rasterized(True)    
    ax1 = fig.add_subplot(2,3,1)
    if FACE_ON: plot_face(ax1, coord[face_mask], density_rad*UnitNumberDensity, n_bins, halfbox, box_range, 1e-4, 10000, log=True)
    else: plot_edge(ax1, coord[edge_mask], density[edge_mask]*UnitNumberDensity, n_bins, halfbox, box_range, 1e-4, 1, log=True)
    density_mesh = ax1.collections[0]
    density_mesh.set_cmap("magma")
    cbar = plt.colorbar(density_mesh, ax = ax1, pad=0.02)  
    cbar.set_label(r'Density [log($\rm cm^{-3}$)]', fontsize=13)
    labels = [1e-5*(10**(x)) for x in range(1,10)]
    cbar.set_ticks(labels)
    cbar.set_ticklabels([round(np.log10(label)) for label in (labels)])
    background_rect = patches.Rectangle((0, 0.80), width=1, height=0.2, color='black', alpha=0.25, transform=ax1.transAxes, fill=True)
    ax1.add_patch(background_rect)
    ax1.text(0.01, 0.96,"t = %0.3f Myr" % times, transform=ax1.transAxes, color="white", fontname='serif', fontsize=12)
    ax1.text(0.01, 0.91,r'LMC/M82 Disk - CIE+PIE', transform=ax1.transAxes, color="white", fontname='serif', fontsize=12)
    ax1.text(0.03, 0.87,r"- $\beta = $" + str(M_load) +  r", $\alpha =$" + str(E_load) + r", $Z_{disk}= " + str(int(InitDiskMetallicity/z_solar)) + r"Z_\odot$", transform=ax1.transAxes, color="white", fontname='serif', fontsize=12)
    ax1.text(0.03, 0.83,r"- $\dot{M}_{SFR} =  " + str(int(sfr)) +  r"M_\odot \, yr^{-1}$, " + r" $R_{inject} =$ " + str(int(R*1000)) + r"pc", transform=ax1.transAxes, color="white", fontname='serif', fontsize=12)
    
    # 2D VELOCITY CENTER VORONOI SLICE 
    ax2 = fig.add_subplot(2,3,2)
    if FACE_ON: plot_face(ax2, coord[face_mask], tan_velocity[face_mask], n_bins, halfbox, box_range, 0, 200, log=False)
    else: plot_edge(ax2, coord[edge_mask], radial_velocity_spherical[edge_mask], n_bins, halfbox, box_range, -100, 1400, log=False)
    velocity_mesh = ax2.collections[0]
    velocity_mesh.set_cmap("viridis")
    cbar = plt.colorbar(velocity_mesh, ax = ax2, pad=0.02)  
    if FACE_ON: cbar.set_label(r'Tangential Velocity [km/s]', fontsize=13)
    else: cbar.set_label(r'Radial Velocity [km/s]', fontsize=13)

    # 2D TEMPERATURE CENTER VORONOI SLICE 
    ax3 = fig.add_subplot(2,3,3)
    if FACE_ON: plot_face(ax3, coord[face_mask], temperature[face_mask], n_bins, halfbox, box_range, 1e3, 1e7, log=True)
    else: plot_edge(ax3, coord[edge_mask], temperature[edge_mask], n_bins, halfbox, box_range, 1e3, 1e7, log=True)
    T_mesh = ax3.collections[0]
    T_mesh.set_cmap("plasma")
    # ax3.axhline(50 - 0.5, linestyle="dashed", color="white", linewidth=1)
    # ax3.axhline(50 + 0.5 , linestyle="dashed", color="white", linewidth=1)
    # ax3.axline((50, 50), slope=np.tan((90 - 60)*np.pi/180), linewidth=1, linestyle="dashed", color="white")
    # ax3.axline((50, 50), slope=-np.tan((90 - 60)*np.pi/180), linewidth=1, linestyle="dashed", color="white")

    cbar3 = plt.colorbar(T_mesh, ax = ax3, pad=0.02) 
    cbar3.set_label(r'Temperature [log(K)]', fontsize=13)

    labels = [1*(10**(x)) for x in range(3,8)]
    cbar3.set_ticks(labels)
    cbar3.set_ticklabels([int(np.log10(label)) for label in labels]) 

    # arc = Arc((50,50), width=3, height=3, angle=0, theta1=30, theta2=90, color="white", linestyle="dotted", linewidth=2)
    # plt.plot(x1,x2, color="white", linestyle="dotted")
    # ax3.add_patch(arc)

    # ax3.text(50 + 0.5, 50 + 1.8, "$60^{\circ}$", color="white", fontsize="medium")
    # circle_bi = patches.Circle((50,50), radius=30, color="white", linestyle="solid", linewidth=2, fill=False)
    # ax3.add_patch(circle_bi)

    # DENSITY RADIAL PROFILE
    ax4 = fig.add_subplot(2,3,4)
    med_d, n_edge, _ = stats.binned_statistic(radius[angular_region], density[angular_region], bins=n_bins, statistic="median")
    hista_nd, n_edge, _ = stats.binned_statistic(radius[angular_region], density[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 84))
    histb_nd, n_edge, _ = stats.binned_statistic(radius[angular_region], density[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 16))

    if FACE_ON:
        ax4.set_xlabel("Radial Distance [kpc]", fontsize=13)
        density_stat, r_edge_d, _ = stats.binned_statistic(r_face, density_rad, bins = 200, range=(0, 5))
        ax4.semilogy(r_edge_d[:-1], density_stat*UnitNumberDensity, label="t = $%0.1f$ Myr" % times, color='midnightblue') 
        if T0_PLOT: ax4.semilogy(rho_init[:,0], rho_init[:,1], label="t = 0.0 Myr", linestyle="dashed", color='midnightblue') 
    else:   
        ax4.set_xlabel("Radius [kpc]", fontsize=13)
        rho_z, z_rho, _ = stats.binned_statistic(r_z, density_z, bins=z_binning, statistic="median")
        ax4.semilogy(z_rho[:-1], rho_z*UnitNumberDensity, color='midnightblue', label = r"Z-axis - t = $%0.1f$ Myr" % times) 
        ax4.semilogy(n_edge[:-1], med_d*UnitNumberDensity, color='green', label = r"Bicone - t = $%0.1f$ Myr" % times) 
        ax4.fill_between(n_edge[:-1], hista_nd*UnitNumberDensity, histb_nd*UnitNumberDensity, color="green", alpha=0.1)

        if T0_PLOT: ax4.semilogy(rho_init[:,0], rho_init[:,1], label="Z-axis - t = 0.0 Myr", linestyle="dashed", color='midnightblue') 
        if CC85_PLOTS: ax4.plot(r_an, rho_n, label='CC85' % times, color='crimson') 

    ax4.set(xlim=(0, upper_x), ylim=(1e-5,1e5))    
    ax4.set_ylabel("Density [$cm^{-3}$]", fontsize=13)
    ax4.legend(loc='upper right', fontsize=13)

    # VELOCITY RADIAL PROFILE 
    ax5 = fig.add_subplot(2,3,5)
    med_v, v_edge, _ = stats.binned_statistic(radius[angular_region], radial_velocity_spherical[angular_region],   bins=n_bins, statistic="median")
    hista_v, v_edge, _ = stats.binned_statistic(radius[angular_region], radial_velocity_spherical[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 84))
    histb_v, v_edge, _ = stats.binned_statistic(radius[angular_region], radial_velocity_spherical[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 16))

    if FACE_ON:
        v_r, r_v, _ = stats.binned_statistic(r_face, radial_velocity[face_mask], bins=200, range=(0, 5))
        v_t, r_t, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], bins=n_bins, range=(0, 5))
        ax5.plot(r_v[:-1], v_r, label="Radial t = $%0.1f$ Myr" % times, color='midnightblue') 
        ax5.plot(r_t[:-1], v_t, label="Circular t = $%0.1f$ Myr" % times, color='red') #
        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='Radial t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if T0_PLOT: ax5.plot(tv_init[:,0], tv_init[:,1], label="Circular t = 0.0 Myr" % times, color='crimson', linestyle="dashed")
        ax5.set_xlabel("Radial Distance [kpc]", fontsize=13)
        ax5.set_ylim(-20, 220) # For a relaxed disk that is settled into equilibrium, the radial velocity should be around 0.
    else:
        v_r, r_v, _ = stats.binned_statistic(r_z, rv_z, bins=z_binning, statistic="median")

        ax5.plot(r_v[:-1], v_r, label='Z-axis - t = $%0.1f$ Myr' % times, color='midnightblue') 
        ax5.plot(v_edge[:-1], med_v, label='Bicone - t = $%0.1f$ Myr' % times, color='green') 
        ax5.fill_between(v_edge[:-1], hista_v, histb_v, color="green", alpha=0.1)

        if T0_PLOT: ax5.plot(vr_init[:,0], vr_init[:,1], label='Z-axis - t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if CC85_PLOTS: ax5.plot(r_an, v_an, label='CC85' % times, color='crimson') 
        ax5.set_ylim(-10, 1500)
        ax5.set_xlabel("Radius [kpc]", fontsize=13)
    ax5.set_xlim(0, upper_x)
    if FACE_ON: ax5.set_ylabel("Velocity [km/s]", fontsize=13)
    else: ax5.set_ylabel("Radial Velocity [km/s]", fontsize=13)
    ax5.legend(loc='upper right', fontsize=13)

    # TEMPERATURE RADIAL PROFILE
    ax6 = fig.add_subplot(2,3,6) 
    med_T, t_edge, _ = stats.binned_statistic(radius[angular_region], temperature[angular_region], bins=n_bins, statistic="median")
    hista_T, t_edge, _ = stats.binned_statistic(radius[angular_region], temperature[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 84))
    histb_T, t_edge, _ = stats.binned_statistic(radius[angular_region], temperature[angular_region], bins=n_bins, statistic=lambda x: np.percentile(x, 16))

    if FACE_ON:
        T, r, _ = stats.binned_statistic(r_face, temperature[face_mask], bins=200, range=(0, 5))
        ax6.semilogy(r[:-1], T, label="t = $%0.1f$ Myr" % times, color='midnightblue')
        if T0_PLOT: ax6.semilogy(T_init[:,0], T_init[:,1], label='t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        ax6.set_xlabel("Radial Distance [kpc]", fontsize=13)
        ax6.set_ylim(1e3,5e7)
    else:
        T, r_T, _ = stats.binned_statistic(r_z, temp_z, bins=z_binning, statistic="median")
        ax6.semilogy(r_T[:-1], T, color='midnightblue', label='Z-axis - t = $%0.1f$ Myr' % times)
        ax6.semilogy(t_edge[:-1], med_T, color='green', label='Bicone - t = $%0.1f$ Myr' % times)
        ax6.fill_between(t_edge[:-1], hista_T, histb_T, color="green", alpha=0.1)

        if T0_PLOT: ax6.semilogy(T_init[:,0], T_init[:,1], label='Z-axis - t = 0.0 Myr', color='midnightblue', linestyle="dashed")
        if CC85_PLOTS: ax6.semilogy(r_an, temp_an, label='CC85', color='red') 
        ax6.set_xlabel("Radius [kpc]")
        ax6.set_ylim(1e3,1e8)
    ax6.set_xlim(0, upper_x)
    ax6.legend(loc="upper right", fontsize=13)
    ax6.set_ylabel("Temperature [K]", fontsize=13)

    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax3.tick_params(axis='both', which='major', labelsize=11)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax5.tick_params(axis='both', which='major', labelsize=11)
    ax6.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout(w_pad=0.00, h_pad=0.00)

    # SAVING THE IMAGES FOR TIMESTEP t 
    if EXTENDED:
        if FACE_ON: img_name = "extended_face_t" + "%0.5f" % t
        else: img_name = "extended_edge_t" + "%0.5f" % t
    else:
        if FACE_ON: img_name = "face_t" + "%0.5f" % t
        else: img_name = "edge_t" + "%0.5f" % t
    print("generating image for time: ", str(t))
    plt.savefig(simulation_directory + img_name + ".png", dpi=150, bbox_inches='tight') 

end = time.time()
print("elapsed time: ", end - start)