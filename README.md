# Raph's thesis scripts
Contains the scripts and figures/videos used to generate plots for my thesis project. the scripts include:
- **avg_radial_plots.py**: Profiles the last 4 (though this can easily be increased or decreased as needed) slices of the temperatures, velocity, and density across different runs. 
- **cooling_rate_comparisons.py**: Generates a side by side comparsion of the volumetric cooling rate between two different snapshots. 
- **create.py**: Generates the IC file that consists of just the hot background.
- **create_disk.py**: Generates the IC file that consists of a cool spinning galactic disk that is embedded in a hot background. Has options for the inclusion of a background grid
- **disk_analytic_plots.py**
- **histograms_mass_volume.py**: Debug plots to make sure that the refinements were that we made to show the distribution of mass and volume at the start and end the simulation
- **ic_prerun_plots.py**: diagnostic plots of the initial conditions. Calculates and plots scatter, slices, and line plots for temperature, density, and velocity 
- **initial_profile_plots.py**: Plots the profiles of velocity, density, and temperature along the central x-y plane and z axis. Used to verify initial conditions.
- **mesh_evolution.py**: Generates the face-on and edge-on voronoi meshes of each snapshot in the simulation.
- **mesh_comp.py**: Plots and compares the voronoi diagrams between two snapshots. Used to compare static vs moving comparisons, but can be generalized to other comparisons
- **phase_diagram_T_nd.py**: Generates a phase diagram temperature as a function of number density, weighted by mass
- **radial_slice_plots.py**: Generates slice plots for density, velocity, and temperature for either an edge or the face-on configuration, along with their respective profiles
- **relax_profiles.py**: Profiles the rotation curves (ie. tangential/circular velocities), density, and temperature at t = 0.0 Gr, t = 1.0 Gyr, and t = 1.5 Gyr across the disk relaxation process.
- **proj_c_T.py, proj_c_v.py, proj_column_density.py**: Used to conduct a case study of a cool cloud
- **time_evolution_plots.py**: Takes every ten snapshots in the simulation and creates plots for velocity, density, and temperature.
- **time_radial_hist.py**: Creates 2D edge-on slice plots velocity, density, and temperature 5, 15, 30 Myrs.
- **voronoi_cooling_plots.py**: Generates edge-on slices and profiles for volumetric cooling rate, metallicity, and electron abundances. 
- **Notebooks**:
    - dynamics_comps.ipynb: Used for generating radial profiles, phase diagrams, and edge-on slices to each other. 
    - time_series_analysis.ipynb: Used to look at anything that requires parsing through a large span in time
    - outflow_rate_comp.ipynb: Computes and compares the mass, energy, and momentum outflow rates between a set of simulations. 
    - voronoi_slices.ipynb: Used whenever a slice plot (and only a slice plot) is needed
    - utilities.py: utility functions that are imported in.<br />

The figures are found in the figures_videos folder, which is divided into subfolders, consisting of:
- **disk_relaxation**: Contains the plots and videos of the M82 disk across the relaxation process.
- **outflow_cooling**: Figures and videos that are related to radiatively cooled winds. 
- **resolution_tests**: Resolution convergence tests for both the static and moving mesh cases.
- **spherically_symmetric_winds**: the videos and plots generated for the configurations used. Note that NFW_75_radial_evo.mp4, NFW_75_radial_evo.mp4, SG_NFW_radial_evo.mp4
- **winds_galactic_disks**: Figures and videos for adiabatic wind simulations that include a galactic disk.


