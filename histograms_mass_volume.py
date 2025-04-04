'''
    Generates histograms of the mass and the volume of the first and last snapshots in the simulation,
    though this can be generalized to any number of snapshots
'''

import h5py
import numpy as np    
import time
import glob
import matplotlib.pyplot as plt
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
n_bins = 301 # general number of bins for the histograms. Some value <= cells_per dim

def mean_molecular_weight(x_e):
    return (4/(1+3*HYDROGEN_MASS_FRACTION + 4*HYDROGEN_MASS_FRACTION*x_e)) * PROTON_MASS_GRAMS

# Equation for temperature - taken from the TNG project website
def Temp_S(x_e, ie):
    return (gamma - 1) * ie/kb * (UnitEnergy_in_cgs/UnitMass_in_g)*mean_molecular_weight(x_e)

######### SIMULATION DATA #########
data = {}

center_boxsize = 10
snaps = [0, 100] # pick which ever snapshots you want to iterate through. I always go with the first and last snapshots

avg_vel = np.zeros(shape=(len(snaps), n_bins))
avg_density = np.zeros(shape=(len(snaps), n_bins))
avg_pressure = np.zeros(shape=(len(snaps), n_bins))
avg_temperature = np.zeros(shape=(len(snaps), n_bins))

for snap in snaps:
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
    # t += 1.5 # offset to time, in order to indicate that my initial conditions started runnning at 1.5 Gyrs
    mvol = boxsize * boxsize * boxsize /len(masses)

    start = time.time()

    ''' Get the radius of the box'''
    rad_x = x_coord - 0.5*boxsize
    rad_y = y_coord - 0.5*boxsize
    rad_z = z_coord - 0.5*boxsize
    radius = np.sqrt(rad_x**2+rad_y**2+rad_z**2) 
    radial_coord = np.sqrt(rad_x**2 + rad_y**2)

    rad_mp = radial_coord[np.isclose(rad_z, 0)] 
    mass_mp = masses[np.isclose(rad_z, 0)] 

    print("Making Histogram")
    volume = masses/density 
    cell_size = 2 * np.cbrt(volume* 3/(4*np.pi))  # the cell size is defined to 2*cell radius = cell diameter

    # Instead taking of ax.loglog, take the np.log10 of the values, and then take the histogram -> using the constant bin size(label as the xaxis as log solar masses or whatever value). 
    masses_log10 = np.log10(masses)

    m, m_edge, _ = stats.binned_statistic(radial_coord, masses, bins=n_bins*3, statistic='mean')
    x = np.abs(m_edge - 0.8)
    inside_box = (np.abs(rad_x) < 5.0) & (np.abs(rad_y) <  5.0) & (np.abs(rad_z) <  5.0)
    inside_disk = (radial_coord <= 2.4) & (np.abs(rad_z) <= 0.25)
    
    radial_diffs = radial_coord - 0.8
    m_rs = masses[np.where(radial_diffs == np.min(radial_diffs))]
    radial_diffs2 = (np.abs(radial_coord - 1.6))
    m_rs2 = masses[np.where(radial_diffs2 == np.min(radial_diffs2))]

    c, c_edge, _ = stats.binned_statistic(radius, cell_size, bins=n_bins*3, statistic='mean')

    print("Histograms completed. Process took: ", time.time() - start, " seconds.")
    print("Plotting")
    ax1.hist(HYDROGEN_MASS_FRACTION)
    times = t*1000

    reference_mass = 450
    refine_masslog = np.log10(2.0 * reference_mass)
    derefine_masslog = np.log10(reference_mass * 0.5)

    ax1.hist(masses_log10, log=True, bins=n_bins, label="t = %.01f Myrs" % times, color="midnightblue")

    cs_log10 = np.log10(cell_size)
    ax2.hist(cs_log10, log=True, bins=n_bins, label="t = %.01f Myrs" % times, color="midnightblue")

    m0f = 0.5*(m_edge[1:] + m_edge[:-1])
    c0f = 0.5*(c_edge[1:] + c_edge[:-1])

    V_ref = 2*3.5e-4 
    cell_ref = 2 * np.cbrt(V_ref* 3/(4*np.pi))  # the cell size is defined to 2*cell radius = cell diameter
   
    V_deref =0
    cell_deref = 2 * np.cbrt(V_deref* 3/(4*np.pi))  # the cell size is defined to 2*cell radius = cell diameter
   

    times = t
    ax3.loglog(m0f, m, label="t = %.01f Gyrs" % times,  color="midnightblue")
    ax4.semilogy(c0f, c, label="t = %.01f Gyrs" % times,  color="midnightblue")

    ax1.set(ylabel="# Cells", xlabel=r"Masses [$log_{10}(M_\odot)$]" )
    ax1.axvline(refine_masslog, linestyle="dashed", color="black",label=r"$log_{10}(2*M_{target})$") # Mass of refinement
    ax1.axvline(derefine_masslog, linestyle="dashed", color="darkgrey",label=r"$log_{10}(0.5*M_{target})$") # Mass of derefinement
    ax1.axvline(np.log10(np.mean(masses[inside_box])), linestyle="dashed", color="crimson", label=r"$log_{10}(\langle M_{inside} \rangle)$")
    ax1.axvline(np.log10(np.mean(masses[inside_disk])), linestyle="dashed", color="orange", label=r"$log_{10}(\langle M_{disk} \rangle)$")


    ax2.set(ylabel="# Cells", xlabel="Cell Size [$log_{10}(kpc)$]", xlim=(np.min(cs_log10), np.max(cs_log10)))

    ax2.axvline(np.log10(cell_ref), linestyle="dashed", color="black",label="$log_{10}(2.0*V_{max})$") # Volume of refinement
    ax2.axvline(np.log10(cell_deref), linestyle="dashed", color="darkgrey",label="$log_{10}(0.5*V_{min})$") # Volume of Derefinement
    ax2.axvline(np.log10(np.mean(cell_size[inside_box])), linestyle="dashed", color="crimson", label=r"$log_{10}(\langle cell_{inside} \rangle)$")
    ax2.axvline(np.log10(np.mean(cell_size[inside_disk])), linestyle="dashed", color="orange", label=r"$log_{10}(\langle cell_{disk} \rangle)$")
    # ax2.axvline(mvol, linestyle="dashed", color="teal", label=r"Mean $V_{sim}$")


    ax3.set(xlabel="Radial Coordinate [kpc]", ylabel="Masses [$M_\odot$]", xlim=(0.1,50), ylim=(1e-1, 1e5))
    
    ax3.axhline(2.0*reference_mass, linestyle="dashed", color="black",label="2.0*$M_{target}$") # Mass of refinement
    ax3.axhline(0.5*reference_mass, linestyle="dashed", color="darkgrey",label="0.5*$M_{target}$") # Mass of derefinement
    ax3.axhline( np.mean(masses[inside_box]), linestyle="dashed", color="crimson", label="Mean $M_{inside}$")
    ax3.axhline(np.mean(masses[inside_disk]), linestyle="dashed", color="orange", label="Mean $M_{disk}$")
    outer = len(masses) - len(masses[inside_box])
    ax3.text(1, 4e4, r"$n_{inner} = %0.02e$" % len(masses[inside_box]) + ", $n_{outer} = %0.02e$" %  outer)



    ax4.axhline(cell_ref, linestyle="dashed", color="black",label=r"$cell_{ref}$") # Volume of refinement
    ax4.axhline(cell_deref, linestyle="dashed", color="darkgrey",label=r"$cell_{deref}$") # Cell size regime of derefinement
    ax4.axhline(np.mean(cell_size[inside_box]), linestyle="dashed", color="crimson", label=r"Mean $cell_{inside}$")
    ax4.axhline(np.mean(cell_size[inside_disk]), linestyle="dashed", color="orange", label=r"Mean $cell_{disk}$")
    ax4.axhline(np.mean(cell_size), linestyle="dashed", color="teal", label=r"Mean $cell_{sim}$")

    # ax4.text(1.1e-1, 2e-6 , "Mean $\Delta x_{box} = %0.03e$ kpc" % np.cbrt(np.mean(volume[inside_box])) + ", $V_{box} = %0.03e$" % np.mean(volume[inside_box]))
    # ax4.text(1.1e-1, 4e-6 , "Mean $\Delta x_{disk} = %0.03e$ kpc" % np.cbrt(np.mean(volume[inside_disk])) +  ", $V_{disk} = %0.03e$" % np.mean(volume[inside_disk]))
    ax4.set(xlabel="Radius [kpc]", ylabel=r"Cell Size [$\rm kpc$]", xlim=(0.1,50), ylim=(1e-2, 5))

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper left")
    plt.savefig("mvhist" + str(t) + "Gyrs_mref.png")
    print("Plotting Completed, Process Took: ", time.time() - start, " seconds.")
