'''
    Defines the functions and constants that are used in the jupyter-notebooks.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter
from scipy import interpolate

HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS or erg/K
GRAVITIONAL_CONSTANT_IN_CGS = 6.6738e-8
HUBBLE = 3.2407789e-18
M_PI = 3.14159265358979323846
seconds_in_yrs = 3.154e+7

##### UNITS #####
UnitVelocity_in_cm_per_s = 1e5 # 10 km/sec 
UnitLength_in_cm = 3.085678e21 # 1 kpc
UnitMass_in_g  = 1.989e33 # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg

G_code = GRAVITIONAL_CONSTANT_IN_CGS/(pow(UnitLength_in_cm,3) * pow(UnitMass_in_g, -1) * pow(UnitTime_in_s, -2)) 
Hubble_code = HUBBLE * UnitTime_in_s # All.Hubble in Arepo

##### FUNCTIONS  #####
### Equations ###
## Potentials and Energies ## 
## Options for NF
def enclosed_mass(R, NFW_M200, NFW_Eps, NFW_C): # concentration of the NFW potential, 10 for M82): # NFW_Eps can either be 0.01 or 0.05
    fac = 1
    R200 = pow( NFW_M200 * G_code / (100 * Hubble_code * Hubble_code), 1.0 / 3) 
    Dc = 200.0 / 3 * (NFW_C * NFW_C * NFW_C) / (np.log(1 + NFW_C) - NFW_C/(1 + NFW_C))
    RhoCrit = 3 * Hubble_code * Hubble_code / (8 * M_PI * G_code) 
    Rs = R200 / NFW_C
    R = np.minimum(R, Rs * NFW_C)
    return fac * 4 * M_PI * RhoCrit * Dc * (-(Rs * Rs * Rs * (1 - NFW_Eps + np.log(Rs) - 2 * NFW_Eps * np.log(Rs) + NFW_Eps * NFW_Eps * np.log(NFW_Eps * Rs))) / ((NFW_Eps - 1) * (NFW_Eps - 1)) + (Rs * Rs * Rs * (Rs - NFW_Eps * Rs - (2 * NFW_Eps - 1) * (R + Rs) * np.log(R + Rs) + NFW_Eps * NFW_Eps * (R + Rs) * np.log(R + NFW_Eps * Rs))) /((NFW_Eps - 1) * (NFW_Eps - 1) * (R + Rs)))

def stellar_potential(z, radial, Mstars, Rstars, zstars_in_UnitLength): 
    return - G_code*(Mstars)/(np.sqrt(pow(radial,2) + pow(Rstars + np.sqrt(pow(z, 2) + pow(zstars_in_UnitLength,2)) , 2)))

def NFW_DM_halo_potential_v2(NFW_M200, radius, NFW_Eps, NFW_C):
    m = enclosed_mass(NFW_M200, NFW_Eps, radius, NFW_C)
    return -(G_code * m) / radius 

## Analytic solutions for the CC85 model ##
# Interior solution
def sol_in(M, r, R):
    T1 = ((3*gamma + 1/M**2)/(1+3*gamma))**(-(3*gamma+1)/(5*gamma+1))
    T2 = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma+1)/(2*(5*gamma+1)))
    return T1*T2 - r/R

# Solution outside the injection radius
##### Taken from Chevalier and Clegg 85
def sol_out(M, r, R):
    T = ((gamma - 1 + 2/M**2)/(1 + gamma))**((gamma + 1)/(2*(gamma - 1)))
    result = M**(2/(gamma - 1))*T - (r/R)**2
    return result

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

### Plots ###
def make_voronoi_slice_edge(gas_xyz, gas_values, image_num_pixels, image_y_value, image_xz_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)  # declare an interpolator of coordinates and values associated with the coordinates
    s = image_xz_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    zs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)

    X,Z = np.meshgrid(xs,zs)
    M_coords = np.transpose(np.vstack([X.ravel(), np.full(len(X.ravel()), image_y_value), Z.ravel()]))
    result = np.transpose(interp(M_coords).reshape(len(xs), len(zs)))

    return result, np.array(xs), np.array(zs)

def make_voronoi_slice_face(gas_xyz, gas_values, image_num_pixels, image_z_value, image_xy_max): 
    interp = interpolate.NearestNDInterpolator(gas_xyz, gas_values)
    s = image_xy_max/image_num_pixels
    xs = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s)
    ys = np.arange(np.min(gas_xyz), np.max(gas_xyz)+s, s) 

    X,Y = np.meshgrid(xs,ys)
    M_coords = np.transpose(np.vstack([X.ravel(), Y.ravel(), np.full(len(X.ravel()), image_z_value)] ))
    result = np.transpose(interp(M_coords).reshape(len(ys), len(xs)))

    return result, np.array(xs), np.array(ys)

def radial_mass_flux(dr, vs, m):
    return 1/dr * np.sum(vs*m)

def calculate_fluxes_mass(H, dr, axis):
    return np.sum(H, axis)/dr/(UnitTime_in_s/seconds_in_yrs) 

def calculate_fluxes_energy(H, dr, axis):
    return np.sum(H, axis)/dr/(UnitTime_in_s) * UnitEnergy_in_cgs 

def get_statistics_mass(H, H_cold, H_r, Hr_cold, dr):
    return calculate_fluxes_mass(H, dr, axis=(1,2,3)), calculate_fluxes_mass(H_cold, dr, axis=(1,2,3)), calculate_fluxes_mass(H_r, dr, axis=(2,3)) , calculate_fluxes_mass(H_r, dr, axis=(1,3)), calculate_fluxes_mass(H_r, dr, axis=(1,2)), calculate_fluxes_mass(Hr_cold, dr, axis=(1,3)), calculate_fluxes_mass(Hr_cold, dr, axis=(1, 2))

def get_statistics_energy(H, H_cold, H_r, Hr_cold, dr):
    return calculate_fluxes_energy(H, dr, axis=(1,2,3)), calculate_fluxes_energy(H_cold, dr, axis=(1,2,3)), calculate_fluxes_energy(H_r, dr, axis=(2,3)) , calculate_fluxes_energy(H_r, dr, axis=(1,3)), calculate_fluxes_energy(H_r, dr, axis=(1,2)), calculate_fluxes_energy(Hr_cold, dr, axis=(1,3)), calculate_fluxes_energy(Hr_cold, dr, axis=(1, 2))

def calculate_std(flux_snaps): 
    f_snaps = np.stack(flux_snaps, axis=0)
    return np.percentile(f_snaps, 16, axis=0), np.percentile(f_snaps, 84, axis=0)   # shape: (len(r_analysis), n_bins)

def plot_fluxes(ax, x_axis, fluxes, std_b, std_a, labeling, coloring,lining, shading):
    ax.plot(x_axis, fluxes, label=labeling, linestyle=lining, color=coloring)
    if shading: ax.fill_between(x_axis, std_b, std_a, color=coloring,  alpha=0.25)

def plot_histogram(y_bins, x_bins, stat_bin, cmap, ax, cbins, x_label, y_label, i, log):
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    X, Y = np.meshgrid(x_bins, y_centers)
    if log: ax.pcolormesh(X, Y, stat_bin.T, cmap=cmap, shading='auto', norm=colors.LogNorm(vmin=cbins[0], vmax=cbins[1]))
    else: ax.pcolormesh(X, Y, stat_bin.T, cmap=cmap, shading='auto', vmin=cbins[0], vmax=cbins[1])

    ax.tick_params(axis='both', which='major', labelsize=11.0)
    if i % 2 == 0: ax.set_ylabel(y_label, fontsize="large")
    else: plt.setp(ax.get_yticklabels(), visible=False)

    if (i == 0) or (i == 1): # if top rpw
        plt.setp(ax.get_xticklabels(), visible=False)
    if (i == 2) or (i == 3): # if bottom row
        ax.set_xlabel(x_label, fontsize="large")


# ### Note: These likely will work, but I haven't tested these out yet. 
# def plot_face(ax, coordinates, value, bins, center, boxsize, minimum, maximum, histb_l, histb_h, log):
#     stat, x_edge, y_edge = make_voronoi_slice_face(coordinates, value, bins, center, boxsize)
#     ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
#     face_mesh = ax.pcolormesh(x_edge, y_edge, stat.T, shading='auto')
#     if (log): face_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
#     else: face_mesh.set_clim(minimum, maximum)

#     def custom_tick_labels(x, pos):
#         return f"{x - boxsize/2:.0f}"
#     ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
#     ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
#     ax.set_xlabel('X [kpc]')
#     ax.set_ylabel('Y [kpc]')

# def plot_edge(ax, coordinates, value, bins, center, boxsize, minimum, maximum, histb_l, histb_h, log):
#     stat, x_edge, z_edge = make_voronoi_slice_edge(coordinates, value, bins, center,  boxsize)
#     ax.set(xlim=(histb_l, histb_h), ylim=(histb_l, histb_h)) 
#     edge_mesh = ax.pcolormesh(x_edge, z_edge, stat.T, shading='auto')
#     if (log): edge_mesh.set_norm(colors.LogNorm(vmin=minimum, vmax=maximum))
#     else: edge_mesh.set_clim(minimum, maximum)

#     def custom_tick_labels(x, pos):
#         return f"{x - boxsize/2:.0f}"
#     ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
#     ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
#     ax.set_xlabel('X [kpc]')
#     ax.set_ylabel('Z [kpc]')