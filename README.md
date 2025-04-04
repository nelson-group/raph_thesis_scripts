# raph_project
Contains many of the scripts used to generate plots and initial conditions for my simulations. Some scripts are more up to date than others. Includes:
- **avg_radial_plots.py**: Profiles the last 5 slices of the temperatures, velocity, and density across different runs. 
- **create.py**: Generates the IC file that consists of just the hot background.
- **create_disk.py**: Generates the IC file that consists of a cool spinning galactic disk that is embedded in a hot background. Has options for the inclusion of a background grid
- **initial_profile_plots.py**: Plots the profiles of velocity, density, and temperature along the central x-y plane and z axis. Used to verify initial conditions.
- **mesh_evolution.py**: Generates the voronoi mesh of each snapshot in the simulation.
- **radial_slice_plots.py**: Generates the slice plots for the density, velocity, and temperature either the edge or the face on configuration. In my outputs, this is commonly refered to as "radial_plots.py". Currently the code struggles with generating profiles on a moving mesh. 
- **time_evolution_plots.py**: Takes every ten snapshots in the simulation and creates plots for velocity, density, and temperature.
- **time_radial_hist.py**: Creates 2D histograms of velocity, density, and temperature at 5, 15, 25, and 35 Myrs.
