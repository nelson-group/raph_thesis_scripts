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
from matplotlib.ticker import MaxNLocator
from scipy import stats

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS


### PARAMETER CONSTANTS ###
filename = "./snap_000.hdf5" 
with h5py.File(filename,'r') as f:
    parameters = dict(f['Parameters'].attrs)
    cells_per_dim = int(np.cbrt(len(f['PartType0']['Density'][()])))

UnitVelocity_in_cm_per_s = parameters["UnitVelocity_in_cm_per_s"] # 1 km/s
UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # 3.08568e+16 seconds 
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2) # 1.9889999999999999e+43 erg
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2) # 6.769911178294542e-22 barye

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 200 # general number of bins for the histograms. Some value <= cells_per dim

def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

######### SIMULATION DATA #########
data = {}
times = np.array([])
v_rm = np.array([])
M = []

legends = []
colors = []
linestyles = []
center_boxsize = 10
files = glob.glob('./snap_*.hdf5')

snaps = [0, 100]

avg_vel = np.zeros(shape=(len(snaps), n_bins))
avg_density = np.zeros(shape=(len(snaps), n_bins))
avg_pressure = np.zeros(shape=(len(snaps), n_bins))
avg_temperature = np.zeros(shape=(len(snaps), n_bins))

zstars_in_UnitLength = 0.15
for i, snap in enumerate(snaps):
    filename = "./snap_%03d.hdf5" % snap
    with h5py.File(filename,'r') as f:
        for key in f['PartType0']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
        parameters = dict(f['Parameters'].attrs)

    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    boxsize = parameters["BoxSize"] # boxsize in kpc
    dx = boxsize/cells_per_dim
    coord = np.transpose(data["Coordinates"])
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    masses = data["Masses"] 
    pressures = data["Pressure"] 
    t = header["Time"]

    mvol = boxsize * boxsize * boxsize /len(masses)


    ''' Get the radius of the box'''
    rad_x = x_coord - 0.5*boxsize
    rad_y = y_coord - 0.5*boxsize
    rad_z = z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)

    rad_mp = radial_coord[np.where(np.isclose(rad_z, 0))] 
    mass_mp = masses[np.where(np.isclose(rad_z, 0))] 

    print("Making Histogram")
    mc = masses[np.argsort(radial_coord)]
    rad_c = np.sort(radial_coord)
    binning = np.linspace(0, 100, 1000)
    # histo,edges = np.histogram(mc, bins=binning)
    rc = np.sort(radius)
    volume = masses/density

    vc = volume[np.argsort(radius)]
    m, m_edge, _ = stats.binned_statistic(radial_coord, masses, bins=n_bins*3, statistic='mean')
    x = np.abs(m_edge - 0.8)
    inside_box = (np.abs(rad_x) < 4.98) & (np.abs(rad_y) <  4.98) & (np.abs(rad_z) <  4.98)
    inside_disk = (radial_coord <= 2.4) & (np.abs(rad_z) <= 0.25)
    
    print(np.unique(x_coord))    
    radial_diffs = radial_coord - 0.8
    m_rs = masses[np.where(radial_diffs == np.min(radial_diffs))]
    radial_diffs2 = (np.abs(radial_coord - 1.6))
    m_rs2 = masses[np.where(radial_diffs2 == np.min(radial_diffs2))]
    print("Mean M at rs = 1.6 kpc", np.mean(m_rs2))

    # print("M at 2*rs = 1.6 kpc", masses[idx2])
    v, v_edge, _ = stats.binned_statistic(radius, volume, bins=n_bins*3, statistic='mean')

    print("Plotting")
    ax1.hist(HYDROGEN_MASS_FRACTION)
    times = t*1000
    ax1.hist(masses, log=True,bins=100, label="t = %.01f Myrs" % times, color="midnightblue")
    ax2.hist(volume, log=True, bins=100, label="t = %.01f Myrs" % times, color="midnightblue")
    ax2.set_xscale("log")

    m0f = 0.5*(m_edge[1:] + m_edge[:-1])
    v0f = 0.5*(v_edge[1:] + v_edge[:-1])
    times = t
    ax3.loglog(m0f, m, label="t = %.01f Gyrs" % times,  color="midnightblue")

    ax4.loglog(v0f, v, label="t = %.01f Gyrs" % times,  color="midnightblue")
    
    # Get the gradient: 
    # ax4.plot(v0f, np.gradient(v, v0f), label="Volume Gradient, t = %.01f Gyrs" % t, color="black", linestyle="dotted")
    # print(np.mean(np.gradient(v)[np.where(v0f <= 8)]))

    ax1.set(ylabel="# Cells", xlabel="Masses [$M_\odot$]")
    # ax1.axvline(x=np.mean(m_rs), linestyle="dashed", color="lightgrey", label="$M_{r_s}$")
    # ax1.axvline(x=np.mean(m_rs2), linestyle="dashed", color="slategrey",  label="$M_{2r_s}$")

    ax1.axvline(2.0 * 450 * 1, linestyle="dashed", color="black",label="2*$M_{target}$") # Mass of refinement
    ax1.axvline(0.5 * 450 * 1, linestyle="dashed", color="darkgrey",label="0.5*$M_{target}$") # Mass of derefinement
    ax1.axvline( np.mean(masses[inside_box]), linestyle="dashed", color="crimson", label="Mean $M_{inside}$")
    ax1.axvline(np.mean(masses[inside_disk]), linestyle="dashed", color="orange", label="Mean $M_{disk}$")
    ax1.set_xscale("log")
    ax2.set(ylabel="# Cells", xlabel="Volume [$kpc^3$]")
    # ax2.axvline(mvol*0.1, linestyle="dashed", color="lightgrey",label="$0.1 \cdot$ Mean Volume")
    ax2.axvline(2*4.0e-5, linestyle="dashed", color="black",label="2*$V_{max}$(Unused)") # Volume of refinement
    ax2.axvline(3.6e-05/2, linestyle="dashed", color="darkgrey",label="0.5*$V_{min}$(Unused)") # Volume of Derefinement
    ax2.axvline( np.mean(volume[inside_box]), linestyle="dashed", color="crimson", label="Mean $V_{inside}$")
    ax2.axvline(np.mean(volume[inside_disk]), linestyle="dashed", color="orange", label="Mean $V_{disk}$")

    # ax3.axvline(3.6e-05, linestyle="dashed", colors)


    # ax3.axhline(np.mean(m_rs), linestyle="dashed", color="black", label="Mass at $r_s=0.8 kpc$")
    # ax3.axvline(x=0.8, linestyle="dashed", color="black")
    # ax3.axvline(x=1.6, linestyle="dashed", color="darkgray", label="Mass at $2 r_s=1.6 kpc$")
    # ax3.axhline(np.mean(m_rs2), linestyle="dashed", color="darkgray")

    # ax3.text(1.8, 800, "Mean $M_{Rs} = %0.1f M_\odot,$" % np.mean(m_rs) + "Mean $M_{2Rs} = %0.1f M_\odot$" % np.mean(m_rs2) )
    ax3.set(xlabel="Radial Coordinate [kpc]", ylabel="Masses [$M_\odot$]", xlim=(0.1,50), ylim=(1e-1, 1e5))

    # All.TargetGasMass = All.TargetGasMassFactor * All.ReferenceGasPartMass
    # P[i].Mass > 2.0 * All.TargetGasMass)
    ax3.axhline(2.0 * 450 * 1, linestyle="dashed", color="black",label="2*$M_{target}$") # Mass of refinement
    ax3.axhline(0.5 * 450 * 1, linestyle="dashed", color="darkgrey",label="0.5*$M_{target}$") # Mass of derefinement
    ax3.axhline( np.mean(masses[inside_box]), linestyle="dashed", color="crimson", label="Mean $M_{inside}$")
    ax3.axhline(np.mean(masses[inside_disk]), linestyle="dashed", color="orange", label="Mean $M_{disk}$")
    outer = len(masses) - len(masses[inside_box])
    ax3.text(1, 4e4, r"$n_{inner} = %0.02e$" % len(masses[inside_box]) + ", $n_{outer} = %0.02e$" %  outer)
                                         
    ax4.axhline(2*4.0e-5, linestyle="dashed", color="black",label="2*$V_{max}$(Unused)") # Volume of refinement
    ax4.axhline(0, linestyle="dashed", color="darkgrey",label="0.5*$V_{min}$(Unused)") # Volume of Derefinement
    ax4.axhline( np.mean(volume[inside_box]), linestyle="dashed", color="crimson", label="Mean $V_{inside}$")
    ax4.axhline(np.mean(volume[inside_disk]), linestyle="dashed", color="orange", label="Mean $V_{disk}$")


    ax4.text(1.1e-1, 2e-6 , "Mean $\Delta x_{box} = %0.03e$ kpc" % np.cbrt(np.mean(volume[inside_box])) + ", $V_{box} = %0.03e$" % np.mean(volume[inside_box]))
    ax4.text(1.1e-1, 4e-6 , "Mean $\Delta x_{disk} = %0.03e$ kpc" % np.cbrt(np.mean(volume[inside_disk])) +  ", $V_{disk} = %0.03e$" % np.mean(volume[inside_disk]))
    ax4.set(xlabel="Radius [kpc]", ylabel="Volume [$kpc^3$]", xlim=(0.1,50), ylim=(1e-6,1))


    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper left")
    plt.savefig("mvhist" + str(t) + "Gyrs_mref.png")

import matplotlib.colors as colors
from scipy import stats
from scipy import interpolate
import matplotlib as mpl
from scipy.spatial import Voronoi, voronoi_plot_2d
import time 
from matplotlib.ticker import FuncFormatter
mpl.rcParams['agg.path.chunksize'] = 10000 # cell overflow fix

### PHYSICAL CONSTANTS ###
HYDROGEN_MASS_FRACTION = 0.76
PROTON_MASS_GRAMS = 1.67262192e-24 # mass of proton in grams
gamma = 5/3
kb = 1.3807e-16 # Boltzmann Constant in CGS

#### Configuration Options ####
FACE_ON = False
T0_PLOT = True 
################################

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
n_bins = 301 # general number of bins for the histograms.
deviation = 5.5
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

center_boxsize = 10
dx = center_boxsize/cells_per_dim


######### SIMULATION DATA #########
start = time.time()
data = {}
files = glob.glob('./snap_*.hdf5')
for i in np.arange(31,  len(files), 2): # select the snapshot range to go through
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
    rad_x = x_coord - 0.5*boxsize
    rad_y = y_coord - 0.5*boxsize
    rad_z = z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    midpoint = boxsize/2

    midpoint = boxsize/2
    lower_bound = midpoint - dx/3
    upper_bound = midpoint + dx/3
 
    face_mask = (z_coord >=lower_bound) & (z_coord <= upper_bound)
    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) 
    z_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) & (x_coord >=lower_bound) & (x_coord <= upper_bound)  


    xy_points = np.vstack((x_coord[face_mask], y_coord[face_mask])).T
    xz_points = np.vstack((x_coord[edge_mask], z_coord[edge_mask])).T

    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(1,2,1)

    rho_xy = density[face_mask]
    # if len(xy_points) > 90000:
    #     xy_points = xy_points[np.random.choice(len(xy_points), size=90000, replace=False)]
    #     rho_xy = density[face_mask][np.random.choice(len(density[face_mask]), size=90000, replace=False)]
    
    vor_xy = Voronoi(xy_points)

    voronoi_plot_2d(vor_xy, ax=ax1, show_points=False, show_vertices=False, line_width=0.25)
    cell_scatter_xy = ax1.scatter(xy_points[:,0], xy_points[:,1], c=rho_xy*UnitNumberDensity, s=0.15, norm=colors.LogNorm(vmin=1e-5, vmax=1e2))
    cbar = plt.colorbar(cell_scatter_xy, ax = ax1, label='Density [log($cm^{-3}$)]')
    ax1.set(xlim=(boxsize/2 - deviation, boxsize/2 + deviation), ylim=(boxsize/2 - deviation, boxsize/2 + deviation), xlabel='X [kpc]', ylabel='Y [kpc]') 
    ax1.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax1.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    # ax1.text(0.01, 0.95, f"Time: {times:.1f} Myrs", transform=ax1.transAxes, fontsize=8)

    ax2 = fig.add_subplot(1,2,2)

    rho_xz = density[edge_mask]
    # if len(xy_points) > 10000:
    #     xz_points = xz_points[np.random.choice(len(xz_points), size=10000, replace=False)]
    #     rho_xz = density[edge_mask][np.random.choice(len(density[edge_mask]), size=10000, replace=False)]
    
    vor_xz = Voronoi(xz_points)



    voronoi_plot_2d(vor_xz, ax=ax2, show_points=False, show_vertices=False, line_width=0.25)
    cell_scatter_z = ax2.scatter(xz_points[:,0], xz_points[:,1], c=rho_xz*UnitNumberDensity, s=0.15, norm=colors.LogNorm(vmin=1e-5, vmax=1e2))
    cbar = plt.colorbar(cell_scatter_z, ax = ax2, label='Density [log($cm^{-3}$)]')
    ax2.set(xlim=(boxsize/2 - deviation, boxsize/2 + deviation), ylim=(boxsize/2 - deviation, boxsize/2 + deviation), xlabel='X [kpc]', ylabel='Z [kpc]') 
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
    ax2.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

    # SAVING THE IMAGES FOR TIMESTEP t 
    img_name = "voronoi_evo" + "%0.5f" % t
    print("generating image for time: ", str(t))
    simulation_directory = str(sys.argv[1]) 

    plt.savefig(simulation_directory + img_name + ".png", dpi=150, bbox_inches='tight') 
    plt.show()

end = time.time()
print("elapsed time: ", end - start)