# Data and code to reproduce figures from the following manuscript
Caccamo, A., Dunstan, D.M., Richardson, M.P., Shaw, A.D., Goodfellow, M. (2025). Dynamics-Informed Priors (DIP) for Neural Mass Modelling. bioRxiv. doi: https://doi.org/10.1101/2025.09.26.678721.

### Prerequisites
The spm12 toolbox is required, please download it from https://www.fil.ion.ucl.ac.uk/spm/.

Scripts to create figures and execute all data contained here are:
run_plot_spectra_params.m
run_plot_data_models.m in run_plot_data_models
run_efficiency_plots.m in run_efficiency
run_reduced_150_generations.m in run_for_15_gen
run_full_model_comparison.m in run_full_model_29_params
run_scz_ctl_dataset.m in run_validate_on_scz

Please note that the DIP_DCM toolbox can be found here: https://github.com/AlessiaCaccamo/DIP_DCM_25

## Citing and using the code
If you use the DIP-DCM toolbox, please cite the above paper. The DIP-DCM toolbox code is open source, under the terms of the GNU General Public License. This works on Windows, Linux, and macOS with an installed version of SPM. The code was originally used with SPM12 only. Code includes third-party functions (SPM, https://www.fil.ion.ucl.ac.uk/spm), with their respective copyright. 
