% Alessia Caccamo, UoE, May 2025
% Bayesian model comparison between full and reduced model (supplementary info)

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_full_model_29_params'


freq_bins = 2:0.5:45;
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Spectra')
load('log_PL_spectra.mat'); 

% raw data is not provided due to storage capacity. 


load('F_all_reduced.mat', 'F_pre_PL_reduced', 'F_post_PL_reduced', 'F_pre_LTG_reduced', 'F_post_LTG_reduced', 'F_pre_LEV_reduced', 'F_post_LEV_reduced');
load('F_all_full.mat', 'F_pre_PL_full', 'F_post_PL_full', 'F_pre_LTG_full', 'F_post_LTG_full', 'F_pre_LEV_full', 'F_post_LEV_full');

%% BMC
% ========Pre-PL=================
% Considering that all 500 DIP modelshave the same evidence, prob=1/500
% log-sum-exp, marginalise over the 500 generative models
% log of the average of the exponentiated log-likelihoods
maxF_pre_PL_reduced = max(F_pre_PL_reduced);
logZ_pre_PL_reduced = log(sum(exp(F_pre_PL_reduced)));

maxF_pre_PL_full = max(F_pre_PL_full);
logZ_pre_PL_full = log(sum(exp(F_pre_PL_full)));

% BMC
logBF_pre_PL = logZ_pre_PL_reduced - logZ_pre_PL_full;


% ========Post-PL=================
maxF_post_PL_reduced = max(F_post_PL_reduced);
logZ_post_PL_reduced = log(sum(exp(F_post_PL_reduced)));

maxF_post_PL_full = max(F_post_PL_full);
logZ_post_PL_full = log(sum(exp(F_post_PL_full)));

% BMC
logBF_post_PL = logZ_post_PL_reduced - logZ_post_PL_full;

% ========Pre-LTG=================
maxF_pre_LTG_reduced = max(F_pre_LTG_reduced);
logZ_pre_LTG_reduced = log(sum(exp(F_pre_LTG_reduced)));

maxF_pre_LTG_full = max(F_pre_LTG_full);
logZ_pre_LTG_full = log(sum(exp(F_pre_LTG_full)));

% BMC
logBF_pre_LTG = logZ_pre_LTG_reduced - logZ_pre_LTG_full;


% ========Post-LTG=================
maxF_post_LTG_reduced = max(F_post_LTG_reduced);
logZ_post_LTG_reduced = log(sum(exp(F_post_LTG_reduced)));

maxF_post_LTG_full = max(F_post_LTG_full);
logZ_post_LTG_full = log(sum(exp(F_post_LTG_full)));

% BMC
logBF_post_LTG = logZ_post_LTG_reduced - logZ_post_LTG_full;

% ========Pre-LEV=================
maxF_pre_LEV_reduced = max(F_pre_LEV_reduced);
logZ_pre_LEV_reduced = log(sum(exp(F_pre_LEV_reduced)));

maxF_pre_LEV_full = max(F_pre_LEV_full);
logZ_pre_LEV_full = log(sum(exp(F_pre_LEV_full)));

% BMC
logBF_pre_LEV = logZ_pre_LEV_reduced - logZ_pre_LEV_full;


% ========Post-LEV=================
maxF_post_LEV_reduced = max(F_post_LEV_reduced);
logZ_post_LEV_reduced = log(sum(exp(F_post_LEV_reduced)));

maxF_post_LEV_full = max(F_post_LEV_full);
logZ_post_LEV_full = log(sum(exp(F_post_LEV_full)));

% BMC
logBF_post_LEV = logZ_post_LEV_reduced - logZ_post_LEV_full;





