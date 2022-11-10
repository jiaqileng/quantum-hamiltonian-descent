## Figure 3: The Three-Phase Picture of QHD.

0. Set MAIN_DIR in the config file.
1. Run `generate_three_phase_data.py` and it will generate the all the plot data. The data are computed for the Levy function.
2. Run `three_phase_plot.py` and it will generate the plot. The plots (png, svg) are saved to `/figures/`.
3. Run `lowE_subspace_plot.py` and it will generate the heatmaps of the first 3 eigen-states of QHD at t = 0 and t = 10. The plots (png, eps) are saved to `/figures/`.