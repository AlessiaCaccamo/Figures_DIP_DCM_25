%% Full model 

clearvars
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/spm12/spm12')

freq_bins = 2:0.5:45;
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Spectra'
load('grand_PL_spectrum.mat'); %"average_across_subjects_pre_PL","average_across_subjects_post_PL");
load('grand_LTG_spectrum.mat'); %"average_across_subjects_pre_LTG","average_across_subjects_post_LTG");
load('grand_LEV_spectrum.mat'); %"average_across_subjects_pre_LEV","average_across_subjects_post_LEV");

load('log_PL_spectra.mat'); % data_psd_pre_PL, data_psd_post_PL
load('log_LTG_spectra.mat');
load('log_LEV_spectra.mat');

load('grand_LTG_subject_spectra.mat');
load('grand_LEV_subject_spectra.mat');
load('grand_PL_subject_spectra.mat');


% %% Load hybrid data for reduced model
% load('models_PL_500_hybrid.mat'); % 'model_pre_PL_all', 'model_post_PL_all'
% load('models_LTG_500_hybrid.mat'); % 'model_pre_LTG_all', 'model_post_LTG_all'
% load('models_LEV_500_hybrid.mat'); % 'model_pre_LEV_all', 'model_post_LEV_all'

%% Load hybrid data for full model
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_full_model_29_params'
% PL
load('models_PL_500_hybrid_29_params.mat')
load('params_PL_hybrid.mat')
load('log_params_PL_hybrid_posteriors.mat')

%LTG
load('models_LTG_500_hybrid_29_params.mat')
load('params_LTG_hybrid.mat')
load('log_params_LTG_hybrid_posteriors.mat')

%LEV
load('models_LEV_500_hybrid_29_params.mat')
load('params_LEV_hybrid.mat')
load('log_params_LEV_hybrid_posteriors.mat')

%% Plot spectra

figure;
subplot(2,3,1);
hold on; 
plot(freq_bins(1:57), mean(model_pre_PL_all,2), 'LineWidth', 1, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), mean(model_post_PL_all,2), 'LineWidth', 1, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_PL, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log Power Spectral Density', 'Fontsize', 14);
legend('Hybrid Pre-PL', 'Data Pre-PL', 'Hybrid Post-PL', 'Data Post-PL'); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);


% % Lamotrigine
subplot(2,3,2);
hold on; 
plot(freq_bins(1:57), mean(model_pre_LTG_all,2), 'LineWidth', 1, 'Color', 'blue'); 
hold on; 
plot(freq_bins(1:57), data_psd_pre_LTG, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
hold on; 
plot(freq_bins(1:57), mean(model_post_LTG_all,2), 'LineWidth', 1, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LTG, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log Power Spectral Density', 'Fontsize', 14);
legend('Hybrid Pre-LTG', 'Data Pre-LTG', 'Hybrid Post-LTG', 'Data Post-LTG');

% % Levetiracetam
subplot(2,3,3);
hold on; 
plot(freq_bins(1:57), mean(model_pre_LEV_all,2), 'LineWidth', 1, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_LEV, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
hold on; 
plot(freq_bins(1:57), mean(model_post_LEV_all,2), 'LineWidth', 1, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LEV, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log Power Spectral Density', 'Fontsize', 14);
ylim([-6 -2]);
legend('Hybrid Pre-LEV', 'Data Pre-LEV', 'Hybrid Post-LEV', 'Data Post-LEV');




