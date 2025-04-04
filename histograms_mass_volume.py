'''
    Generates histograms of the mass and the volume of the first and last snapshots in the simulation,
    though this can be easily generalized to any number of snapshots
'''

import h5py
import numpy as np    
import matplotlib.pyplot as plt
from scipy import stats

#### FUNCTIONS ####
def calculate_cell_size(volume):
    return 2 * np.cbrt(volume * 3 /(4 * np.pi))

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
n_bins = 301 # general number of bins for the histograms. Some value <= cells_per dim
center_boxsize = 10

######### REFINEMENT PARAMETERS #########
reference_mass = 450
refine_masslog10 = np.log10(2.0 * reference_mass)
derefine_masslog10 = np.log10(reference_mass * 0.5)
V_ref = 2*2e-4 
# V_deref = 0 
cell_ref = 2 * np.cbrt(V_ref)  
# cell_deref = calculate_cell_size(V_ref) # N

######### SIMULATION DATA #########
snaps = [0, 100] # pick which ever snapshots you want to iterate through. I always go with the first and last snapshots
for snap in snaps:
    data = {}
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    print("Reading from snaphot_%0.03d.hdf5" % snap)
    filename = "./snap_%03d.hdf5" % snap
    with h5py.File(filename,'r') as f:
        for key in f['PartType0']:
            data[key] = f['PartType0'][key][()]
        header = dict(f['Header'].attrs)
        parameters = dict(f['Parameters'].attrs)
    boxsize = parameters["BoxSize"] # boxsize in kpc
    dx = boxsize/cells_per_dim
    x_coord = data["Coordinates"][:,0] 
    y_coord = data["Coordinates"][:,1]
    z_coord = data["Coordinates"][:,2]
    density = data["Density"]
    internal_energy = data["InternalEnergy"] # NOTE: This is specific internal energy, not the actual internal energy
    masses = data["Masses"] 
    masses_log10 = np.log10(masses)
    pressures = data["Pressure"] 
    t = header["Time"]
    volume = masses/density 
    mvol = boxsize * boxsize * boxsize /len(masses)
    cell_size = calculate_cell_size(volume) # the cell size is defined to 2*cell radius = cell diameter
    cs_log10 = np.log10(cell_size)

    rad_x, rad_y, rad_z = x_coord - 0.5*boxsize, y_coord - 0.5*boxsize, z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)
    inside_box = (np.abs(rad_x) < 5.0) & (np.abs(rad_y) <  5.0) & (np.abs(rad_z) <  5.0)
    inside_disk = (radial_coord <= 2.4) & (np.abs(rad_z) <= 0.3)

    mean_mass_interior = np.mean(masses[inside_box])
    mean_mass_disk = np.mean(masses[inside_disk])
    mean_cell_interior = np.mean(cell_size[inside_box])
    mean_cell_disk = np.mean(cell_size[inside_disk])

    m_profile, m_edge, _ = stats.binned_statistic(radial_coord, masses, bins=900, statistic='mean')
    cell_profile, c_edge, _ = stats.binned_statistic(radius, cell_size, bins=900, statistic='mean')
    m0f = 0.5*(m_edge[1:] + m_edge[:-1])
    c0f = 0.5*(c_edge[1:] + c_edge[:-1])

    times = t*1000
    print("Making Histograms")
    ax1.hist(masses_log10, log=True, bins=n_bins, label="t = %.01f Myrs" % times, color="midnightblue")
    ax1.axvline(refine_masslog10, linestyle="dashed", color="black",label=r"$log_{10}(2*M_{target})$") # Mass of refinement
    ax1.axvline(derefine_masslog10, linestyle="dashed", color="darkgrey",label=r"$log_{10}(0.5*M_{target})$") # Mass of derefinement
    ax1.axvline(np.log10(mean_mass_interior), linestyle="dashed", color="crimson", label=r"$log_{10}(\langle M_{inside} \rangle)$")
    ax1.axvline(np.log10(mean_mass_disk), linestyle="dashed", color="orange", label=r"$log_{10}(\langle M_{disk} \rangle)$")
    ax1.set(ylabel="# Cells", xlabel=r"Masses [$log_{10}(M_\odot)$]" , xlim=(np.min(masses_log10), np.max(masses_log10)))

    ax2.hist(cs_log10, log=True, bins=n_bins, label="t = %.01f Myrs" % times, color="midnightblue")
    ax2.axvline(np.log10(cell_ref), linestyle="dashed", color="black",label="$log_{10}(cell_{ref}$)") # Volume of refinement
    # ax2.axvline(np.log10(cell_deref), linestyle="dashed", color="darkgrey",label="$log_{10}(0.5*V_{min})$") # Volume of Derefinement -> this is 0
    ax2.axvline(np.log10(mean_cell_interior), linestyle="dashed", color="crimson", label=r"$log_{10}(\langle cell_{inside} \rangle)$")
    ax2.axvline(np.log10(mean_cell_disk), linestyle="dashed", color="orange", label=r"$log_{10}(\langle cell_{disk} \rangle)$")
    ax2.axvline(np.log10(np.mean(cell_size)), linestyle="dashed", color="teal", label=r"$log_{10}(\langle cell_{sim} \rangle$)")
    ax2.set(ylabel="# Cells", xlabel="Cell Size [$log_{10}(kpc)$]", xlim=(np.min(cs_log10), np.max(cs_log10)))
    
    print("Making Profiles")
    ax3.loglog(m0f, m_profile, label="t = %.01f Myr" % times,  color="midnightblue")
    ax3.axhline(2.0*reference_mass, linestyle="dashed", color="black",label="2.0*$M_{target}$") # Mass of refinement
    ax3.axhline(0.5*reference_mass, linestyle="dashed", color="darkgrey",label="0.5*$M_{target}$") # Mass of derefinement
    ax3.axhline(mean_mass_interior, linestyle="dashed", color="crimson", label=r"$\langle M_{inside} \rangle$")
    ax3.axhline(mean_mass_disk, linestyle="dashed", color="orange", label=r"$\langle M_{disk} \rangle$")
    ax3.set(xlabel="Radial Coordinate [kpc]", ylabel="Masses [$M_\odot$]", xlim=(0.1,50), ylim=(1, 2e3))

    ax4.semilogy(c0f, cell_profile, label="t = %.01f Myrs" % times,  color="midnightblue")
    ax4.axhline(cell_ref, linestyle="dashed", color="black",label="$cell_{ref}$")# Volume of refinement
    # ax4.axhline(0, linestyle="dashed", color="darkgrey",label=r"$cell_{deref}$") # Cell size regime of derefinement -> this is 0 
    ax4.axhline(np.mean(mean_cell_interior), linestyle="dashed", color="crimson", label=r"$\langle cell_{inside} \rangle$")
    ax4.axhline(np.mean(mean_cell_disk), linestyle="dashed", color="orange", label=r"$\langle cell_{disk} \rangle$")
    ax4.axhline(np.mean(cell_size), linestyle="dashed", color="teal", label=r"$\langle cell_{sim} \rangle$")
    ax4.set(xlabel="Radius [kpc]", ylabel=r"Cell Size [$\rm kpc$]", xlim=(0.1,50), ylim=(1e-2, 1))

    ax1.legend(loc="lower left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="lower left")
    ax4.legend(loc="upper right")

    plt.savefig("mvhist" + str(t) + "Gyrs_mref.png")