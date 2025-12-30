'''
    Generates a static vs moving voronoi mesh comparison
'''
import h5py
import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

def custom_tick_labels(x, pos):
    return f"{x - boxsize/2:.0f}"

data = {}
### PARAMETER CONSTANTS ###
filename = "./output_center_ref/snap_000.hdf5" 
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

UnitLength_in_cm = parameters["UnitLength_in_cm"] # 1 kpc 
UnitMass_in_g = parameters["UnitMass_in_g"] # 1 solar mass
UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3) # 6.76989801444063e-32 g/cm^3
UnitNumberDensity = UnitDensity_in_cgs/PROTON_MASS_GRAMS

boxsize = parameters["BoxSize"] # boxsize in kpc
n_bins = 301 # general number of bins for the histograms.
deviation = 5
histb_l = boxsize/2 - deviation # boundary of histogram - lower bound
histb_h = boxsize/2  + deviation # boundary of histogram - upper bound

middle = boxsize/2
fig = plt.figure(figsize=(9,4))
fig.set_rasterized(True)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


inner_boxsize = 10
dx = inner_boxsize/cells_per_dim
lower_bound = middle - dx*2
upper_bound = middle + dx*2
######### SIMULATION DATA #########
start = time.time()

data = {}
files = ["./output_center_ref/snap_200.hdf5", "./output_static/snap_100.hdf5"]
for i, file in enumerate(files):
    with h5py.File(file,'r') as f:
        for key in f['PartType0']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    t = header["Time"]
    times = t
    rad_x = x_coord - 0.5*boxsize
    rad_y = y_coord - 0.5*boxsize
    rad_z = z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2)
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)

    edge_mask = (y_coord >=lower_bound) & (y_coord <= upper_bound) 

    xz_points = np.vstack((x_coord[edge_mask], z_coord[edge_mask])).T


    rho_xz = density[edge_mask]
    vor_xz = Voronoi(xz_points)

    if file == './output_static/snap_100.hdf5':
        voronoi_plot_2d(vor_xz, ax=ax1, show_points=False, show_vertices=False, line_width=0.05, line_colors='gray')
        cell_scatter_z = ax1.scatter(xz_points[:,0], xz_points[:,1], c=rho_xz*UnitNumberDensity, s=0.001, norm=colors.LogNorm(vmin=1e-4, vmax=10), cmap="magma")
        cbar = plt.colorbar(cell_scatter_z, ax = ax1, pad=0.02)# , label=r'Density [log($\rm cm^{-3}$)]')
        cbar.set_label(r'Density [log($\rm cm^{-3}$)]', fontsize=12)
        ax1.set(xlim=(boxsize/2 - deviation, boxsize/2 + deviation), ylim=(boxsize/2 - deviation, boxsize/2 + deviation), xlabel='X [kpc]', ylabel='Z [kpc]')
        ax1.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        ax1.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))

        labels = [1e-5*(10**(x)) for x in range(1,7)]
        cbar.set_ticks(labels)
        cbar.set_ticklabels([round(np.log10(label)) for label in (labels)])
    else:
        voronoi_plot_2d(vor_xz, ax=ax2, show_points=False, show_vertices=False, line_width=0.15, line_colors='black')
        cell_scatter_z = ax2.scatter(xz_points[:,0], xz_points[:,1], c=rho_xz*UnitNumberDensity, s=0.005, norm=colors.LogNorm(vmin=1e-4, vmax=10), cmap="magma")
        cbar = plt.colorbar(cell_scatter_z, ax = ax2, pad=0.02) # label=r'Density [log($\rm cm^{-3}$)]')
        cbar.set_label(r'Density [log($\rm cm^{-3}$)]', fontsize=12)
        ax2.set(xlim=(boxsize/2 - deviation, boxsize/2 + deviation), ylim=(boxsize/2 - deviation, boxsize/2 + deviation)) # , xlabel='X [kpc]', ylabel='Z [kpc]')
        ax2.set_xlabel('X [kpc]', fontsize=12)
        ax2.set_ylabel('Z [kpc]', fontsize=12)

        ax2.xaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        ax2.yaxis.set_major_formatter(FuncFormatter(custom_tick_labels))
        labels = [1e-5*(10**(x)) for x in range(1,7)]

        cbar.set_ticks(labels)
        cbar.set_ticklabels([round(np.log10(label)) for label in (labels)])

plt.tight_layout(w_pad=0)
# SAVING THE IMAGES FOR TIMESTEP t 
img_name = "voronoi_static_moving"
print("generating image")

plt.savefig(img_name + ".pdf", dpi=150, bbox_inches='tight') 
plt.show()

end = time.time()
print("elapsed time: ", end - start)