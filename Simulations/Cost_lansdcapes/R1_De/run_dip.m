%% Run DIP
addpath('/home/links/ac1376/DEMO_logNoise/spm12')  
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/spectral') 
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/external/fieldtrip/utilities') 
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/dcm_meeg') 
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/Neural_Models')
addpath('/home/links/ac1376/multistart/spm12-maint')      
addpath('/home/links/ac1376/DEMO_logNoise/spm12')
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/spectral')     
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/external/fieldtrip/utilities') 
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/dcm_meeg')     
addpath('/home/links/ac1376/DEMO_logNoise/spm12/spm12/toolbox/Neural_Models')
addpath('DIP_DCM_25_fixed_params')
cd '/home/links/ac1376/Simulations'


%addpath('/Users/alessiacaccamo/Documents/Exeter/Data/DIP_DCM_25')
load('synthetic_psd_params_R1De.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];

freq=2:0.5:30;
total_num=1000;
select_num=500;
    % MODEL_PIPELINE Executes the DIP-DCM modeling pipeline for the LFP NMM with given parameter bounds.
    % INPUTS:
    % total_num   - Total number of GA explorations (e.g., 1000).
    % select_num  - Number of selected priors (e.g., 400).
    % This function allows modelling of two datasets representing experimental conditions to be
    % compared. 
    % set_paths; % Script that adds paths to the folders containing files for loading the data and running the model 
    GA_params_matrices = cell(1, 10); % GA parameter matrices for two datasets
    DCM_totals = cell(1, 10); % Store DCM structures for two datasets in the cell array
    out_totals = cell(1, 10); % Store GA output structures for two datasets in the cell array
for i=1:2 % for each dataset
    data_psd=spec(:,i);  % Load the row vector data_psd and row vector freq_bins 
    freq_bins=freq;
    out_total=cell(total_num,1); % GA output for each repeat (defined by total_num), for each dataset
    for nsim=1:total_num % Run or load GA results for each repeat
    %out=run_lfp_MOGA(data_psd,freq_bins,nsim); % This function allows to run MOGA here, otherwise run on the server and load the files as follows.
    load(['/home/links/ac1376/Simulations/DIP_sim_5/GA_LFP_dataset_' num2str(i) '_nsim_' num2str(nsim) '.mat'], 'out'); 
    out_total{nsim}=out; % Store outputs associated with each repeat in out_total cell array
    end
    out_totals{i} = out_total; % Store all outputs for both datasets in the cell array out_totals
    GA_params_matrices{i} = save_dcm_priors(out_totals{i}, data_psd,freq_bins,total_num, select_num); % Store MOGA-generated parameters, for each repeat and each dataset in the cell array MOGA_params_matrices. This generates DCM priors based on selected, optimal parameter regions. 
    %save(['GA_priors_hybrid_dataset_' num2str(i) '.mat'], 'GA_params_matrices', 'similar_sim_numbers', 'psd_m_all');
    DCM_total=cell(1,select_num); % DCM structures for each selected GA prior.
    for nsim=1:select_num 
    DCM=run_lfp_hybrid(GA_params_matrices{i},nsim,data_psd,freq_bins); % Run a DCM for each of the selected priors. 
    save(['/home/links/ac1376/Simulations/DIP_sim_7/DCM_LFP_dataset_' num2str(i) '_nsim_' num2str(nsim) '.mat'], 'DCM');
    end 
end
