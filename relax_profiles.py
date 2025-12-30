
'''
   Script used to generate the velocity, density, and temperature profiles for the M82, MW, and SMC disks across the relaxation process. 
'''
import h5py
import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import matplotlib as mpl
from matplotlib.lines import Line2D
import time 
from matplotlib.ticker import FuncFormatter
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

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
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 bary
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

# Mean molecular weight based off of an electron abundance - currently x_e = 1, but subject to change in future simulations
def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

t0_files = [ "./Disk_M82/snap_000.hdf5", "./Disk_MW/snap_000.hdf5", "./Disk_SMC/snap_000.hdf5"]
t1_files = ["./Disk_M82/snap_050.hdf5", "./Disk_MW/snap_050.hdf5", "./Disk_SMC/snap_050.hdf5"]
mm_files = ['./M82_mm.hdf5', './MW_mm.hdf5', './SMC_mm.hdf5']

labels = ["M82, t = ", "MW, t = ", "SMC, t = "]
color_map = plt.get_cmap('jet')
colors = color_map(np.linspace(0, 1, len(t0_files)))

### PARAMETER CONSTANTS ###
data = {}
filename = "./Disk_M82/snap_000.hdf5" 
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

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

inner_boxsize = 10
mid_center = inner_boxsize/2
dx = inner_boxsize/cells_per_dim
devx = dx/1e6
middle = boxsize/2

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
######### SIMULATION DATA #########
start = time.time()
data = {}
for i, file in enumerate(t0_files):
    print(file)
    with h5py.File(file,'r') as f:
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
    temperature = Temp_S(1, internal_energy)
    t = header["Time"]
    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    rmax = inner_boxsize/2*np.sqrt(3)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) # max radius is center boxsize/2 *sqrt(3) = 8.66
    face_mask = (np.isclose(z_coord, middle)) & (radial_coord <= inner_boxsize/2*np.sqrt(2)) 
    z_mask = (np.isclose(x_coord, middle)) & (np.isclose(y_coord, middle)) & (np.abs(rad_z) <= inner_boxsize/2*0.98)
    r_face = radial_coord[face_mask] 

    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+devx), vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)

    tvf, tv_edge, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], statistic='mean', bins=n_bins)
    rhof, rho_edge, _ = stats.binned_statistic(r_face, density[face_mask]*UnitNumberDensity, statistic='mean', bins=n_bins)
    Tf, T_edge, _ = stats.binned_statistic(r_face, temperature[face_mask], statistic='mean', bins=n_bins)

    rf = 0.5*(tv_edge[1:] + tv_edge[:-1])
    
    ax1.plot(rf, tvf, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dotted")
    ax2.semilogy(rf, rhof, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dotted")
    ax3.semilogy(rf, Tf,  label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dotted")

for i, file in enumerate(t1_files):
    print(file)
    with h5py.File(file,'r') as f:
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
    temperature = Temp_S(1, internal_energy)
    t = header["Time"]
    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    rmax = inner_boxsize/2*np.sqrt(3)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) # max radius is center boxsize/2 *sqrt(3) = 8.66
    face_mask = (np.isclose(z_coord, middle)) & (radial_coord <= inner_boxsize/2*np.sqrt(2)) 
    z_mask = (np.isclose(x_coord, middle)) & (np.isclose(y_coord, middle)) & (np.abs(rad_z) <= inner_boxsize/2*0.98)
    r_face = radial_coord[face_mask] 

    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+devx), vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)

    tvf, tv_edge, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], statistic='mean', bins=n_bins)
    rhof, rho_edge, _ = stats.binned_statistic(r_face, density[face_mask]*UnitNumberDensity, statistic='mean', bins=n_bins)
    Tf, T_edge, _ = stats.binned_statistic(r_face, temperature[face_mask], statistic='mean', bins=n_bins)

    rf = 0.5*(tv_edge[1:] + tv_edge[:-1])
    
    ax1.plot(rf, tvf, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dashed")
    ax2.semilogy(rf, rhof, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dashed")
    ax3.semilogy(rf, Tf,  label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="dashed")

for i, file in enumerate(mm_files):
    print(file)
    with h5py.File(file,'r') as f:
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
    temperature = Temp_S(1, internal_energy)
    t = header["Time"]
    ''' Get the radial distance of the box'''
    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    rmax = inner_boxsize/2*np.sqrt(3)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2) # max radius is center boxsize/2 *sqrt(3) = 8.66

    lower_bound = middle - dx
    upper_bound = middle + dx 
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound) & (radial_coord <= inner_boxsize/2*np.sqrt(2))
    z_mask = (np.isclose(x_coord, middle)) & (np.isclose(y_coord, middle)) & (np.abs(rad_z) <= inner_boxsize/2*0.98)
    r_face = radial_coord[face_mask] 
    #### Velocities - for the center disk plane face####
    radial_velocity = (vel_x*rad_x + vel_y*rad_y)/(radial_coord + devx) 
    tvx, tvy = vel_x - radial_velocity*rad_x/(radial_coord+devx), vel_y - radial_velocity*rad_y/(radial_coord+devx)
    tan_velocity = np.sqrt(tvx**2 + tvy**2)

    tvf, tv_edge, _ = stats.binned_statistic(r_face, tan_velocity[face_mask], statistic='mean', bins=n_bins)
    rhof, rho_edge, _ = stats.binned_statistic(r_face, density[face_mask]*UnitNumberDensity, statistic='mean', bins=n_bins)
    Tf, T_edge, _ = stats.binned_statistic(r_face, temperature[face_mask], statistic='mean', bins=n_bins)

    rf = 0.5*(tv_edge[1:] + tv_edge[:-1])
    
    ax1.plot(rf, tvf, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="solid")
    ax2.semilogy(rf, rhof, label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="solid")
    ax3.semilogy(rf, Tf,  label = labels[i] + str(t) + " Gyr", color=colors[i], linestyle="solid")

ax1.set(xlim=(0, 7), ylim=(0, 300))
ax2.set(xlim=(0, 7), ylim=(1e-5, 10000))
ax3.set(xlim=(0, 7), ylim=(5e3, 1e7))

ax1.set_xlabel('Radial Distance [kpc]', fontsize=13)
ax2.set_xlabel('Radial Distance [kpc]', fontsize=13)
ax3.set_xlabel('Radial Distance [kpc]', fontsize=13)
ax1.set_ylabel('Circular Velocity [km/s]', fontsize=13)
ax2.set_ylabel(r'Density [$cm^{-3}$]', fontsize=13)
ax3.set_ylabel('Temperature [K]', fontsize=13)


legend_elements = [ Line2D([0], [0], color=colors[0], label='M82'),
                    Line2D([0], [0], color=colors[1], label='MW'),
                    Line2D([0], [0], color=colors[2], label='SMC'),
                    Line2D([0], [0], color="black", linestyle='dotted' ,label='t = 0 Gyr'),
                    Line2D([0], [0], color="black", linestyle='dashed', label='t = 1.0 Gyr'),
                    Line2D([0], [0], color="black",linestyle='solid', label='t = 1.5 Gyr'),
                    ]

ax1.legend(handles=legend_elements, loc='upper right', ncols=2, fontsize=12)
ax2.legend(handles=legend_elements, loc='upper right', ncols=2, fontsize=12)
ax3.legend(handles=legend_elements, loc='upper right', ncols=2, fontsize=12)

plt.tight_layout()

plt.savefig("face_profiles_ic.pdf", dpi=150, bbox_inches='tight')

