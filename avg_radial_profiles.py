'''
    Grabs every output folder in directory, grabs the last n_snaps_avg snapshots and takes the average temperature, velocity, and density
    for each output folder. Can be used for loading parameters, injection radii, resolution, and gravity interactions
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
from scipy import optimize
import csv

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

### SIMULATION CONSTANTS - keep it consistent with Param.txt ###
UnitVelocity_in_cm_per_s = 1e5 # 1 km/s
UnitLength_in_cm = 3.08568e+21 # 1 kiloparsec
UnitMass_in_g = 1.989e+33 # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye

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
    return M**(2/(gamma - 1))*T - (r/R)**2
# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

### INITIAL CONFIGURATION PARAMETERS ###
n_bins = 150

######### SIMULATION DATA #########
data = {}
times = np.array([])
v_rm = np.array([])
M = []
Temperatures = np.array([])

legends = []
colors = []
linestyles = []
n_snaps = 100
n_snaps_avg = 4
outputs = sorted(glob.glob('./load_tests/output_*'))
print(outputs)
cells_per_dim = [600 for output in outputs]
# print(cells_per_dim)
avg_vel = np.zeros(shape=(len(outputs), n_bins))
avg_density = np.zeros(shape=(len(outputs), n_bins))
avg_pressure = np.zeros(shape=(len(outputs), n_bins))
avg_temps = np.zeros(shape=(len(outputs), n_bins))


f_ratio_v = np.zeros(shape=(len(outputs), n_bins))
f_ratio_rho = np.zeros(shape=(len(outputs), n_bins))
f_ratio_p = np.zeros(shape=(len(outputs), n_bins))
f_ratio_t = np.zeros(shape=(len(outputs), n_bins))

analytic_e_v = np.zeros(shape=(len(outputs), n_bins))
analytic_e_rho = np.zeros(shape=(len(outputs), n_bins))
analytic_e_p = np.zeros(shape=(len(outputs), n_bins))
analytic_e_t = np.zeros(shape=(len(outputs), n_bins))

r_bins = np.linspace(0.0, 10, n_bins+1)
r_faces = 0.5*(r_bins[:-1] + r_bins[1:])

alpha_def = 0.25
beta_def = 0.25
x_max = 8

for o, output in enumerate(outputs):
    print(output)
    files = glob.glob('/' + output + '/snap_*.hdf5')
    for i in np.arange(n_snaps - n_snaps_avg, n_snaps):
        filename = output + "/snap_%03d.hdf5" % i
        with h5py.File(filename,'r') as f:
            for key in f['PartType0']:
                data[key] = f['PartType0'][key][()]
            header = dict(f['Header'].attrs)
            parameters = dict(f['Parameters'].attrs)
            # cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()]))) 
            # also get cells in the injection region/(cell size(dx)/R) - relative spatial resolution. -> consider resolution.
        boxsize = parameters["BoxSize"] # boxsize in kpc
        # print(boxsize)
        dx = boxsize/cells_per_dim[o]
        coord = np.transpose(data["Coordinates"])
        x_coord = data["Coordinates"][:,0] 
        y_coord = data["Coordinates"][:,1]
        z_coord = data["Coordinates"][:,2]
        density = data["Density"]
        density_gradient = data["DensityGradient"] 
        internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
        masses = data["Masses"] 
        pressures = data["Pressure"] 
        vel_x = data["Velocities"][:,0]
        vel_y = data["Velocities"][:,1] 
        vel_z = data["Velocities"][:,2] 
        vel = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        E = internal_energy*masses # NOTE: This is the actual internal energy
        t = header["Time"]

        M_load = parameters["M_load"]
        E_load = parameters["E_load"]

        R = parameters["injection_radius"]
        sfr = parameters["sfr"]

        ''' Get the radius of the box'''
        rad_x = x_coord - 0.5*boxsize
        rad_y = y_coord - 0.5*boxsize
        rad_z = z_coord - 0.5*boxsize
        radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
        radial_velocity = (vel_x*rad_x + vel_y*rad_y + vel_z*rad_z)/radius
        calc_temps = Temp_S(1, internal_energy)

        # VELOCITY DATA    
        vel_stat, r_edge_v, bin_n = stats.binned_statistic(radius,  radial_velocity , bins = n_bins, range=[0, boxsize])
        avg_vel[o] += vel_stat
        # DENSITY DATA  
        density_stat, r_edge_d, bin_n = stats.binned_statistic(radius, density, bins = n_bins, range=[0, boxsize])
        avg_density[o] += density_stat*UnitDensity_in_cgs/PROTON_MASS_GRAMS # we multiply because we want to get this in analytic units -> number density per cm^3

        # PRESSURE DATA
        pressure_stat, r_edge_p, bin_n = stats.binned_statistic(radius, pressures, bins = n_bins, range=[0, boxsize])
        avg_pressure[o] += pressure_stat*UnitPressure_in_cgs/kb

        # TEMPERATURE DATA
        temp_stat, r_edge_t, bin_n = stats.binned_statistic(radius, calc_temps, bins = n_bins, range=[0, boxsize])
        avg_temps[o] += temp_stat
        if o == outputs.index('./load_tests/output_default'):
            print(np.median(calc_temps[(radius >= 0.98) & (radius <= 1)]))

    avg_vel[o] /= n_snaps_avg   
    avg_density[o] /= n_snaps_avg
    avg_pressure[o] /= n_snaps_avg
    avg_temps[o] /= n_snaps_avg

    # ANALYTIC CALCULATIONS
    r_an = np.linspace(0.01, boxsize, n_bins)
    r_cm = r_an*UnitLength_in_cm

    r_in = r_an[np.where(r_an <= R)]
    r_out = r_an[np.where(r_an > R)]

    r_in_cm = r_in*UnitLength_in_cm
    r_out_cm = r_out*UnitLength_in_cm
    R_cm = R*UnitLength_in_cm

    s_in_yr = 3.154e+7
    grams_in_M_sun = 1.989e33
    M_dot_wind = sfr*M_load # solar masses per 1 year -> get this in grams per second 
    M_dot_cm = (M_dot_wind*UnitMass_in_g)/s_in_yr # grams/second
    E_dot_wind = E_load*3e41*sfr # this is in ergs/second 

    M_dot_code = M_dot_wind/(UnitMass_in_g/grams_in_M_sun)*(UnitTime_in_s/s_in_yr)
    E_dot_code = E_dot_wind/UnitEnergy_in_cgs*UnitTime_in_s

    M1 = optimize.fsolve(sol_in, x0=np.full(len(r_in), 0.01), args=(r_in))
    M2 = optimize.fsolve(sol_out, x0=np.full(len(r_out), 20), args=(r_out))
    M = np.concatenate([M1, M2])

    v_an = (M*np.sqrt(E_dot_code/M_dot_code)*(((gamma - 1)*M**2 + 2)/(2*(gamma - 1)))**(-0.5)) # this is in code units
    v_in = v_an[np.where(r_an <= R)]
    v_out = v_an[np.where(r_an > R)]
    v_cm = v_an*UnitVelocity_in_cm_per_s

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

    # print("simulation temperature:")
    # print("analytic temp")
    # print(temp

    # DEVIATION 
    analytic_e_v[o] = (avg_vel[o] - v_an)/v_an * 100 
    analytic_e_rho[o] = (avg_density[o] - rho_n)/rho_n * 100 
    analytic_e_p[o] = (avg_pressure[o] - pressure_an)/pressure_an *100 
    analytic_e_t[o] = (avg_temps[o] - temp_an)/temp_an *100 

    # print(analytic_e_v)
    relative_cell_size = dx/R 
    # legends.append(r"$\alpha$: %0.2f$, $\beta$: %0.2f$, R: %0.2f, cells: %i" % (E_load, M_load, R, cells_per_dim))
    legends.append(r"$\alpha$= %0.2f, $\beta$= %0.2f" % (E_load, M_load))
    # legends.append(r"$N_{cells}$: %i, size: %0.3f" % (cells_per_dim[o], relative_cell_size))
    # legends.append(r"$R_{\rm inject}$ = %0.2f kpc" % (R))

    # if "high" in output:
    #     linestyles.append("solid")
    #     colors.append("crimson")
    # if "low" in output:
    #     linestyles.append("solid")
    #     colors.append("teal")
    # if "default" in output:
    #     linestyles.append("solid")
    #     colors.append("black")

    if "beta" in output:
        linestyles.append("dotted")
        if M_load > beta_def:
            colors.append("crimson")
        else:
            colors.append("teal")
    if "alpha" in output:
        linestyles.append("dashed")
        if E_load > alpha_def:
            colors.append("crimson")
        else:
            colors.append("teal")
    if "both" in output:
        linestyles.append("dashdot")
        if E_load > alpha_def:
            colors.append("crimson")
        else:
            colors.append("teal")
    if "default" in output:
        colors.append("black")
        linestyles.append("solid")

    # if "SGNFW" in output:
    #     linestyles.append("dashdot")
    #     colors.append("crimson")
    #     legends.append("SG-NFW")
    # elif "SG" in output:
    #     linestyles.append("dashed")
    #     colors.append("teal")
    #     legends.append("SG")
    # elif "NFW" in output:
    #     linestyles.append("dotted")
    #     colors.append("darkmagenta")
    #     legends.append("NFW")
    # elif "default" in output:
    #     colors.append("black")
    #     linestyles.append("solid")
    #     legends.append("None")
    print(linestyles)
    print(colors)
    print(legends)

f_ratio_v = avg_vel/avg_vel[outputs.index('./load_tests/output_default')]
f_ratio_rho = avg_density/avg_density[outputs.index('./load_tests/output_default')]
f_ratio_p = avg_pressure/avg_pressure[outputs.index('./load_tests/output_default')]
f_ratio_t = avg_temps/avg_temps[outputs.index('./load_tests/output_default')]

fig = plt.figure(figsize=(13, 7)) # default 16, 8. 21, 12 for load tests
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
# ax4 = fig.add_subplot(2,2,4)

ax1.set(xlim=(0, x_max), ylim=(0, 3000))
ax2.set(xlim=(0, x_max), ylim=(1e-5,1))
ax3.set(xlim=(0, x_max), ylim=(5e4,1e8))
# ax4.set(xlim=(0, x_max), ylim=(1e1,1e8))


ratio_fiducial_v = ax1.inset_axes([0, -0.35, 1, 0.30], sharex=ax1)
analytic_error_v = ax1.inset_axes([0, -0.70, 1, 0.30], sharex=ax1)
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ratio_fiducial_v.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ratio_fiducial_rho = ax2.inset_axes([0, -0.35, 1, 0.30], sharex=ax2)
analytic_error_rho = ax2.inset_axes([0, -0.70, 1, 0.30], sharex=ax2)
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ratio_fiducial_rho.tick_params(axis='x', bottom=False, labelbottom=False)

ratio_fiducial_t = ax3.inset_axes([0, -0.35, 1, 0.30], sharex=ax3)
analytic_error_t = ax3.inset_axes([0, -0.70, 1, 0.30], sharex=ax3)
ax3.tick_params(axis='x', bottom=False, labelbottom=False)
ratio_fiducial_t.tick_params(axis='x', bottom=False, labelbottom=False)

# ratio_fiducial_p = ax4.inset_axes([0, -0.35, 1, 0.30], sharex=ax4)
# analytic_error_p = ax4.inset_axes([0, -0.70, 1, 0.30], sharex=ax4)
# ax4.tick_params(axis='x', bottom=False, labelbottom=False)
# ratio_fiducial_p.tick_params(axis='x', bottom=False, labelbottom=False)


analytic_error_v.set_xlabel("Radius [kpc]", fontsize=13)
ax1.set_ylabel(r"Radial Velocity [km/s]", fontsize=13)
ratio_fiducial_v.set_ylabel("Ratio", fontsize=13)
analytic_error_v.set_ylabel("Error [%]", fontsize=13)

analytic_error_rho.set_xlabel(r"Radius [kpc]", fontsize=13)
ax2.set_ylabel(r"Density [$\rm cm^{-3}$]", fontsize=13)
ratio_fiducial_rho.set_ylabel("Ratio", fontsize=13)
analytic_error_rho.set_ylabel("Error [%]", fontsize=13)

analytic_error_t.set_xlabel("Radius [kpc]", fontsize=13)
ax3.set_ylabel(r"Temperature [K]", fontsize=13)
ratio_fiducial_t.set_ylabel("Ratio", fontsize=13)
analytic_error_t.set_ylabel("Error [%]", fontsize=13)

# analytic_error_p.set_xlabel("Radius [kpc]", fontsize=13)
# ax4.set_ylabel(r"Pressure [$\rm K \, cm^{-3}$]", fontsize=13)
# ratio_fiducial_p.set_ylabel("Ratio", fontsize=13)
# analytic_error_p.set_ylabel("Deviation [%]", fontsize=13)

ax1.plot(r_edge_v[:-1], avg_vel[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     
analytic_error_v.plot(r_edge_v[:-1], analytic_e_v[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     

for v, vel in enumerate(avg_vel):
    if v != outputs.index('./load_tests/output_default'):
        # v_sim = ax1.plot(r_edge[:-1], avg_vel[v], label=outputs[v]) 
        ax1.plot(r_edge_v[:-1], avg_vel[v], label=legends[v], color=colors[v], linestyle=linestyles[v])
        analytic_error_v.plot(r_edge_v[:-1], analytic_e_v[v], label=legends[v], color=colors[v], linestyle=linestyles[v])
        ratio_fiducial_v.plot(r_edge_v[:-1], f_ratio_v[v], label=legends[v], color=colors[v], linestyle=linestyles[v])
analytic_error_v.set_ylim(-5,15)
ratio_fiducial_v.set_ylim(0, 3)
ratio_fiducial_v.set_yticks([0, 1, 2])
# analytic_error_v.set_yticks([-5, 0, 5, 10])

ax2.semilogy(r_edge_d[:-1], avg_density[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     
analytic_error_rho.plot(r_edge_d[:-1], analytic_e_rho[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     
for d, rho in enumerate(avg_density):
    if d != outputs.index('./load_tests/output_default'):
    # rho_sim = ax2.semilogy(r_edge[:-1], avg_density[d]*UnitDensity_in_cgs/PROTON_MASS_GRAMS, label=outputs[d])
        ax2.semilogy(r_edge_d[:-1], avg_density[d], label=legends[d], color=colors[d], linestyle=linestyles[d])
        analytic_error_rho.plot(r_edge_d[:-1], analytic_e_rho[d], label=legends[d], color=colors[d], linestyle=linestyles[d])
        ratio_fiducial_rho.plot(r_edge_d[:-1], f_ratio_rho[d], label=legends[d], color=colors[d], linestyle=linestyles[d])
analytic_error_rho.set_ylim(-25,15)
ratio_fiducial_rho.set_ylim(0, 3)
ratio_fiducial_rho.set_yticks([0, 1, 2])
# analytic_error_rho.set_yticks([-20 -10, 0, 10])


ax3.semilogy(r_edge_t[:-1], avg_temps[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     
analytic_error_t.plot(r_edge_t[:-1], analytic_e_t[2], label=legends[2], color=colors[2], linestyle=linestyles[2], zorder=10)     
for t, temps in enumerate(avg_temps):
    if t != outputs.index('./load_tests/output_default'):
        ax3.semilogy(r_edge_t[:-1], avg_temps[t], label=legends[t], color=colors[t], linestyle=linestyles[t])
        analytic_error_t.plot(r_edge_t[:-1], analytic_e_t[t], label=legends[t], color=colors[t], linestyle=linestyles[t])
        analytic_error_t.set_ylim(-20,10)
        ratio_fiducial_t.plot(r_edge_t[:-1], f_ratio_t[t], label=legends[t], color=colors[t], linestyle=linestyles[t])
        print(f_ratio_t[t])
ratio_fiducial_t.set_ylim(0, 3)
ratio_fiducial_t.set_yticks([0, 1, 2])
# analytic_error_t.set_yticks([-20, 10, 0, 10]) # , 15])


# for p, press in enumerate(avg_pressure):
#     p_sim = ax4.semilogy(r_edge_p[:-1], press, label=legends[p], color=colors[p], linestyle=linestyles[p])
#     # ax3.semilogy(r_edge_t[:-1], avg_temps[t], label=legends[t], color=colors[t], linestyle=linestyles[t])
#     analytic_error_p.plot(r_edge_p[:-1], analytic_e_p[p], label=legends[p], color=colors[p], linestyle=linestyles[p])
#     analytic_error_p.set_ylim(-15,15)
#     if t != outputs.index('./load_tests/output_default'):
#         ratio_fiducial_p.plot(r_edge_p[:-1], f_ratio_p[p], label=legends[p], color=colors[p], linestyle=linestyles[p])
#         ratio_fiducial_p.set_ylim(0, 4.5)


analytic_error_v.yaxis.set_label_coords(-0.12, 0.5)
ratio_fiducial_v.yaxis.set_label_coords(-0.12, 0.5)
ax1.yaxis.set_label_coords(-0.12, 0.5)

analytic_error_rho.yaxis.set_label_coords(-0.12, 0.5)
ratio_fiducial_rho.yaxis.set_label_coords(-0.12, 0.5)
ax2.yaxis.set_label_coords(-0.12, 0.5)

analytic_error_t.yaxis.set_label_coords(-0.104, 0.5)
ratio_fiducial_t.yaxis.set_label_coords(-0.104, 0.5)
ax3.yaxis.set_label_coords(-0.104, 0.5)

# analytic_error_p.yaxis.set_label_coords(-0.095, 0.5)
# ratio_fiducial_p.yaxis.set_label_coords(-0.095, 0.5)
# ax4.yaxis.set_label_coords(-0.095, 0.5)

# fig.supxlabel('Distance[kpc]', y=0.055, fontsize=13)

ax1.tick_params(axis='y', which='major', labelsize=11)
ax2.tick_params(axis='y', which='major', labelsize=11)# , length=8, width=1.5)
ax3.tick_params(axis='y', which='major', labelsize=11) # , length=8, width=1.5)

analytic_error_v.tick_params(axis="both", which="major", labelsize=11)
ratio_fiducial_v.tick_params(axis='y', which='major', labelsize=11)

analytic_error_rho.tick_params(axis="both", which="major", labelsize=11)
ratio_fiducial_rho.tick_params(axis='y', which='major', labelsize=11)

analytic_error_t.tick_params(axis="both", which="major", labelsize=11)
ratio_fiducial_t.tick_params(axis='y', which='major', labelsize=11)

l1 = ax1.legend(loc='upper right', fontsize=12)
l1.set_zorder(11)
l2 = ax2.legend(loc='upper right', fontsize=12)
l2.set_zorder(11)
l3 = ax3.legend(loc='upper right', fontsize=12)
l3.set_zorder(11)

# ax4.legend(loc='upper right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.18)

plt.savefig("./moving_mesh_load_tests_plots.pdf",bbox_inches='tight', dpi=150)
plt.show()