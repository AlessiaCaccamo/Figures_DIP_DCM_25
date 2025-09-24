%% Figures - parameters

clearvars
% set path
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25'
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/spm12/spm12')

%% Load empirical data
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

%% Load model spectra GA
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_model_spectra_LEV_grand_final.mat');
psd_m_pre_all_LEV=psd_m_pre_all;
psd_m_post_all_LEV=psd_m_post_all;
load('MOGA_LFP_model_spectra_LTG_grand_final.mat');
psd_m_pre_all_LTG=psd_m_pre_all;
psd_m_post_all_LTG=psd_m_post_all;
load('MOGA_LFP_model_spectra_PL_grand_final.mat');
psd_m_pre_all_PL=psd_m_pre_all;
psd_m_post_all_PL=psd_m_post_all;

%% Load model spectra DCM
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_model_spectra_grand_log_PL_31_May.mat'); % 'model_pre_PL', 'model_post_PL');
load('DCM_LFP_model_spectra_grand_log_LTG_31_May.mat'); % 'model_pre_LTG', 'model_post_LTG');
load('DCM_LFP_model_spectra_grand_log_LEV_31_May.mat'); % 'model_pre_LEV', 'model_post_LEV');
pre_LEV_spec = real(model_pre_LEV{1});
post_LEV_spec=real(model_post_LEV{1});
pre_LTG_spec=real(model_pre_LTG{1});
post_LTG_spec=real(model_post_LTG{1});
pre_PL_spec=real(model_pre_PL{1});
post_PL_spec=real(model_post_PL{1});

%% Load exp params
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_params_final.mat'); % 'all_sim_params_pre_PL', 'all_sim_params_post_PL');
load('MOGA_LFP_LTG_params_final.mat'); % 'all_sim_params_pre_LTG', 'all_sim_params_post_LTG');
load('MOGA_LFP_LEV_params_final.mat'); % 'all_sim_params_pre_LEV', 'all_sim_params_post_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_params.mat'); % 'params_pre_DCM_PL', 'cov_pre_DCM_PL', 'params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_params.mat'); % 'params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_params.mat'); % 'params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'params_post_DCM_LEV', 'cov_post_DCM_LEV');

%% load log params
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_log_params_final.mat'); % 'all_sim_log_params_pre_PL', 'all_sim_log_params_post_PL');
load('MOGA_LFP_LTG_log_params_final.mat'); % 'all_sim_log_params_pre_LTG', 'all_sim_log_params_post_LTG');
load('MOGA_LFP_LEV_log_params_final.mat'); % 'all_sim_log_params_pre_LEV', 'all_sim_log_params_post_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_log_params_final.mat'); % 'log_params_pre_DCM_PL', 'cov_pre_DCM_PL', 'log_params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_log_params_final.mat'); % 'log_params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'log_params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_log_params_final.mat'); % 'log_params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'log_params_post_DCM_LEV', 'cov_post_DCM_LEV');


%% Hybrid DIP-DCM method
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Hybrid'
load('models_PL_500_hybrid.mat', 'model_pre_PL_all', 'model_post_PL_all');
load('models_LTG_500_hybrid.mat', 'model_pre_LTG_all', 'model_post_LTG_all');
load('models_LEV_500_hybrid.mat', 'model_pre_LEV_all', 'model_post_LEV_all');

%%
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25'
% selected GA repeats based on fitness
load('GA_selected_repeats.mat'); % 'similar_sim_numbers'

% how to obtain them
n=1;
for sim = 1:1000
    RMSE_MOGA_LEV_pre(1,sim) = sqrt(mean((data_psd_pre_LEV(n:57) - psd_m_pre_all_LEV(n:57,sim)).^2)); 
    RMSE_MOGA_LEV_post(1,sim) = sqrt(mean((data_psd_post_LEV(n:57) - psd_m_post_all_LEV(n:57,sim)).^2));
    RMSE_MOGA_LTG_pre(1,sim) = sqrt(mean((data_psd_pre_LTG(n:57) - psd_m_pre_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_LTG_post(1,sim) = sqrt(mean((data_psd_post_LTG(n:57) - psd_m_post_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(n:57) - psd_m_pre_all_PL(n:57,sim)).^2));
    RMSE_MOGA_PL_post(1,sim) = sqrt(mean((data_psd_post_PL(n:57) - psd_m_post_all_PL(n:57,sim)).^2));
end
similar_sim_numbers = cell(1, 6);
[RMSE_LEV_pre, indices_LEV_pre] = mink(RMSE_MOGA_LEV_pre, 500); % mink(A)mink(A,k,1) computes the k smallest values in each column of A and
[RMSE_LEV_post, indices_LEV_post] = mink(RMSE_MOGA_LEV_post, 500);
[RMSE_LTG_pre, indices_LTG_pre] = mink(RMSE_MOGA_LTG_pre, 500);
[RMSE_LTG_post, indices_LTG_post] = mink(RMSE_MOGA_LTG_post, 500);
[RMSE_PL_pre, indices_PL_pre] = mink(RMSE_MOGA_PL_pre, 500);
[RMSE_PL_post, indices_PL_post] = mink(RMSE_MOGA_PL_post, 500);
similar_sim_numbers{1} = indices_LEV_pre;
similar_sim_numbers{2} = indices_LEV_post;
similar_sim_numbers{3} = indices_LTG_pre;
similar_sim_numbers{4} = indices_LTG_post;
similar_sim_numbers{5} = indices_PL_pre;
similar_sim_numbers{6} = indices_PL_post;

%% Figure 2B. Mean spectra
% Placebo
% Select spectra
figure;
%figure('Units', 'inches', 'Position', [0, 0, 14, 10]); % 12x8 inches for better spacing
tiledlayout(3,3, 'Padding', 'compact', 'TileSpacing', 'compact'); 

subplot(3,3,1);
hold on; 
plot(freq_bins(1:57), pre_PL_spec, '-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), post_PL_spec,'-', 'LineWidth', 1.5, 'Color', 're'); 
hold on;
plot(freq_bins(1:57), data_psd_post_PL, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
ylim([min([data_psd_pre_PL;data_psd_post_PL;pre_PL_spec;post_PL_spec]) max([data_psd_pre_PL;data_psd_post_PL;pre_PL_spec;post_PL_spec])]);
legend('DCM Pre-PL', 'Data Pre-PL', 'DCM Post-PL', 'Data Post-PL','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
% % Lamotrigine
subplot(3,3,4);
hold on; 
plot(freq_bins(1:57), pre_LTG_spec, '-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on; 
plot(freq_bins(1:57), data_psd_pre_LTG, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
plot(freq_bins(1:57), post_LTG_spec, '-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LTG, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('DCM Pre-LTG', 'Data Pre-LTG', 'DCM Post-LTG', 'Data Post-LTG','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_LTG;data_psd_post_LTG;pre_LTG_spec;post_LTG_spec]) max([data_psd_pre_LTG;data_psd_post_LTG;pre_LTG_spec;post_LTG_spec])]);

% % Levetiracetam
subplot(3,3,7);
hold on; 
plot(freq_bins(1:57), pre_LEV_spec,'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_LEV, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), post_LEV_spec,'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LEV, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
ylim([-6 -2]);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('DCM Pre-LEV', 'Data Pre-LEV', 'DCM Post-LEV', 'Data Post-LEV','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_LEV;data_psd_post_LEV;pre_LEV_spec;post_LEV_spec]) max([data_psd_pre_LEV;data_psd_post_LEV;pre_LEV_spec;post_LEV_spec])]);

subplot(3,3,2);
hold on; 
plot(freq_bins(1:57), mean(psd_m_pre_all_PL(:,similar_sim_numbers{5}),2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), mean(psd_m_post_all_PL(:,similar_sim_numbers{6}),2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_PL, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('GA Pre-PL','Data Pre-PL', 'GA Post-PL', 'Data Post-PL','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_PL;data_psd_post_PL;mean(psd_m_pre_all_PL(:,similar_sim_numbers{5}),2);mean(psd_m_post_all_PL(:,similar_sim_numbers{6}),2)]) max([data_psd_pre_PL;data_psd_post_PL;mean(psd_m_pre_all_PL(:,similar_sim_numbers{5}),2);mean(psd_m_post_all_PL(:,similar_sim_numbers{6}),2)])]);

% % Lamotrigine
subplot(3,3,5);
hold on; 
plot(freq_bins(1:57), mean(psd_m_pre_all_LTG(:,similar_sim_numbers{3}),2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on; 
plot(freq_bins(1:57), data_psd_pre_LTG, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
plot(freq_bins(1:57), mean(psd_m_post_all_LTG(:,similar_sim_numbers{4}),2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LTG, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('GA Pre-LTG', 'Data Pre-LTG', 'GA Post-LTG', 'Data Post-LTG','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_LTG;data_psd_post_LTG;mean(psd_m_pre_all_LTG(:,similar_sim_numbers{3}),2);mean(psd_m_post_all_LTG(:,similar_sim_numbers{4}),2)]) max([data_psd_pre_LTG;data_psd_post_LTG;mean(psd_m_pre_all_LTG(:,similar_sim_numbers{3}),2);mean(psd_m_post_all_LTG(:,similar_sim_numbers{4}),2)])]);

% % Levetiracetam
subplot(3,3,8);
hold on; 
plot(freq_bins(1:57), mean(psd_m_pre_all_LEV(:,similar_sim_numbers{5}),2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_LEV, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), mean(psd_m_post_all_LEV(:,similar_sim_numbers{6}),2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LEV, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
ylim([-6 -2]);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('GA Pre-LEV','Data Pre-LEV', 'GA Post-LEV','Data Post-LEV','fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_LEV;data_psd_post_LEV;mean(psd_m_pre_all_LEV(:,similar_sim_numbers{5}),2);mean(psd_m_post_all_LEV(:,similar_sim_numbers{6}),2)]) max([data_psd_pre_LEV;data_psd_post_LEV;mean(psd_m_pre_all_LEV(:,similar_sim_numbers{5}),2);mean(psd_m_post_all_LEV(:,similar_sim_numbers{6}),2)])]);

subplot(3,3,3);
hold on; 
plot(freq_bins(1:57), mean(model_pre_PL_all,2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on; 
plot(freq_bins(1:57), mean(model_post_PL_all,2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_PL, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('DIP Pre-PL', 'Data Pre-PL', 'DIP Post-PL', 'Data Post-PL', 'fontsize',10); % 'SOGA Post-PL', 'Data Post-PL', 'Fontsize', 12);
ylim([min([data_psd_pre_PL;data_psd_post_PL;mean(model_pre_PL_all,2);mean(model_post_PL_all,2)]) max([data_psd_pre_PL;data_psd_post_PL;mean(model_pre_PL_all,2);mean(model_post_PL_all,2)])]);

% % Lamotrigine
subplot(3,3,6);
hold on; 
plot(freq_bins(1:57), mean(model_pre_LTG_all,2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on; 
plot(freq_bins(1:57), data_psd_pre_LTG, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
hold on; 
plot(freq_bins(1:57), mean(model_post_LTG_all,2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LTG, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('DIP Pre-LTG', 'Data Pre-LTG', 'DIP Post-LTG', 'Data Post-LTG', 'fontsize',10);
ylim([min([data_psd_pre_LTG;data_psd_post_LTG;mean(model_pre_LTG_all,2);mean(model_post_LTG_all,2)]) max([data_psd_pre_LTG;data_psd_post_LTG;mean(model_pre_LTG_all,2);mean(model_post_LTG_all,2)])]);

% % Levetiracetam
subplot(3,3,9);
hold on; 
plot(freq_bins(1:57), mean(model_pre_LEV_all,2),'-', 'LineWidth', 1.5, 'Color', 'blue'); 
hold on;
plot(freq_bins(1:57), data_psd_pre_LEV, '--', 'LineWidth', 1.5, 'Color', 'blue');
hold on;
hold on; 
plot(freq_bins(1:57), mean(model_post_LEV_all,2),'-', 'LineWidth', 1.5, 'Color', 'red'); 
hold on;
plot(freq_bins(1:57), data_psd_post_LEV, '--', 'LineWidth', 1.5, 'Color', 'red');
hold off;
xlabel('Frequency (Hz)', 'Fontsize', 14);
ylabel('Log PSD', 'Fontsize', 14);
ylim([-6 -2]);
set(gca,'FontSize',18, 'FontName', 'Helvetica')
legend('DIP Pre-LEV', 'Data Pre-LEV', 'DIP Post-LEV', 'Data Post-LEV', 'fontsize', 10);
ylim([min([data_psd_pre_LEV;data_psd_post_LEV;mean(model_pre_LEV_all,2);mean(model_post_LEV_all,2)]) max([data_psd_pre_LEV;data_psd_post_LEV;mean(model_pre_LEV_all,2);mean(model_post_LEV_all,2)])]);

% set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, 12, 8]); % Match figure size
% exportgraphics(gcf, 'Figure_3.pdf', 'Resolution', 300, 'ContentType', 'vector');


%% GA Bounds
lb_1 = [-1.4142 -1.4142 -1.4142 -1.4142 -1.1068 -1.0046 -1.3208 -1.0046 -1.0046 -1.0846 -3.2533 -2.8284 -3.2769 -1.0000 -0.7071 -0.3536 -0.5851 -0.3774 -0.3536 -0.3774 -0.3536 -0.3634 -0.3536 -0.3536 -0.4002 -0.5000 -0.6625]; 
ub_1 = [1.4142 1.4142 1.4142 1.4142 1 1 1 1.0509 1.0006 1.1017 2.8284 2.9713 2.8284 1 0.7768 0.3536 0.3536 0.3536 0.3562 0.3536 0.3562 0.3536 0.3712 0.4410 0.3536 0.5000 0.5000];

lb(1) = 1*exp(lb_1(1)); 
lb(2) = 2*exp(lb_1(2));  
lb(3) = 4/1000*exp(lb_1(3)); 
lb(4) = 16/1000*exp(lb_1(4)); 
lb(5) = 8*exp(lb_1(5)); 
lb(6) = 128*exp(lb_1(6)); 
lb(7) = 128*exp(lb_1(7));
lb(8) = 64*exp(lb_1(8));
lb(9) = 64*exp(lb_1(9));
lb(10) = 4*exp(lb_1(10));
lb(11) = 32*exp(lb_1(11));
lb(12) = 16*exp(lb_1(12));
lb(13) = 4*exp(lb_1(13));
lb(14) = 4*exp(lb_1(14))/1000;
lb(15) = 2*exp(lb_1(15))/1000;
lb(16) = exp(lb_1(16));
lb(17) = exp(lb_1(17));
lb(18) = exp(lb_1(18));
lb(19) = exp(lb_1(19));
lb(20) = exp(lb_1(20));
lb(21) = exp(lb_1(21));
lb(22) = exp(lb_1(22));
lb(23) = exp(lb_1(23));
lb(24) = exp(lb_1(24));
lb(25) = exp(lb_1(25));
lb(26) = exp(lb_1(26));
lb(27) = exp(lb_1(27));

ub(1) = 1*exp(ub_1(1)); 
ub(2) = 2*exp(ub_1(2));  
ub(3) = 4/1000*exp(ub_1(3)); 
ub(4) = 16/1000*exp(ub_1(4)); 
ub(5) = 8*exp(ub_1(5)); 
ub(6) = 128*exp(ub_1(6)); 
ub(7) = 128*exp(ub_1(7));
ub(8) = 64*exp(ub_1(8));
ub(9) = 64*exp(ub_1(9));
ub(10) = 4*exp(ub_1(10));
ub(11) = 32*exp(ub_1(11));
ub(12) = 16*exp(ub_1(12));
ub(13) = 4*exp(ub_1(13));
ub(14) = 4*exp(ub_1(14))/1000;
ub(15) = 2*exp(ub_1(15))/1000;
ub(16) = exp(ub_1(16));
ub(17) = exp(ub_1(17));
ub(18) = exp(ub_1(18));
ub(19) = exp(ub_1(19));
ub(20) = exp(ub_1(20));
ub(21) = exp(ub_1(21));
ub(22) = exp(ub_1(22));
ub(23) = exp(ub_1(23));
ub(24) = exp(ub_1(24));
ub(25) = exp(ub_1(25));
ub(26) = exp(ub_1(26));
ub(27) = exp(ub_1(27));

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};


%% 3. load DCM params. 
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_params_final.mat'); % 'all_sim_params_pre_PL', 'all_sim_params_post_PL');
load('MOGA_LFP_LTG_params_final.mat'); % 'all_sim_params_pre_LTG', 'all_sim_params_post_LTG');
load('MOGA_LFP_LEV_params_final.mat'); % 'all_sim_params_pre_LEV', 'all_sim_params_post_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_params.mat'); % 'params_pre_DCM_PL', 'cov_pre_DCM_PL', 'params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_params.mat'); % 'params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_params.mat'); % 'params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'params_post_DCM_LEV', 'cov_post_DCM_LEV');

%% load log params
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_log_params_final.mat'); % 'all_sim_log_params_pre_PL', 'all_sim_log_params_post_PL');
load('MOGA_LFP_LTG_log_params_final.mat'); % 'all_sim_log_params_pre_LTG', 'all_sim_log_params_post_LTG');
load('MOGA_LFP_LEV_log_params_final.mat'); % 'all_sim_log_params_pre_LEV', 'all_sim_log_params_post_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_log_params_final.mat'); % 'log_params_pre_DCM_PL', 'cov_pre_DCM_PL', 'log_params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_log_params_final.mat'); % 'log_params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'log_params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_log_params_final.mat'); % 'log_params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'log_params_post_DCM_LEV', 'cov_post_DCM_LEV');

%% Plot all DCM params. Supplementary 4
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
figure;
positions = 1:15; % Define parameters
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(3,5, idx); %
    hold on;
    log_mean = params_pre_DCM_PL(i, 1); 
    log_variance = cov_pre_DCM_PL(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    hold on;
    log_mean_post = params_post_DCM_PL(i, 1); 
    log_variance_post = cov_post_DCM_PL(i, 1);
    log_sigma_post = sqrt(log_variance_post);
    x_post = linspace(exp(log_mean_post - 4 * log_sigma_post), exp(log_mean_post + 4 * log_sigma_post), 100);
    y_post = lognpdf(x_post, log_mean_post, log_sigma_post);
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'red'); 
    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    hold on;
end
legend('Pre-PL DCM', 'Post-PL DCM', 'Location', 'northwest');

figure;
% LTG
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
hold on;
positions = 1:15 ; % Define the positions for the parameters
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(3,5, idx);
    log_mean = params_pre_DCM_LTG(i, 1); 
    log_variance = cov_pre_DCM_LTG(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    hold on;
    log_mean_post = params_post_DCM_LTG(i, 1); 
    log_variance_post = cov_post_DCM_LTG(i, 1);
    log_sigma_post = sqrt(log_variance_post);
    x_post = linspace(exp(log_mean_post - 4 * log_sigma_post), exp(log_mean_post + 4 * log_sigma_post), 100);
    y_post = lognpdf(x_post, log_mean_post, log_sigma_post);
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'red'); 
    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    hold on;
end
legend('Pre-LTG DCM', 'Post-LTG DCM', 'Location', 'northwest');

%LEV
figure;
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
hold on;
positions = 1:15;
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(3,5, idx);
    log_mean = params_pre_DCM_LEV(i, 1); 
    log_variance = cov_pre_DCM_LEV(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    hold on;
    log_mean_post = params_post_DCM_LEV(i, 1); 
    log_variance_post = cov_post_DCM_LEV(i, 1);
    log_sigma_post = sqrt(log_variance_post);
    x_post = linspace(exp(log_mean_post - 4 * log_sigma_post), exp(log_mean_post + 4 * log_sigma_post), 100);
    y_post = lognpdf(x_post, log_mean_post, log_sigma_post);
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'red'); 
    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    hold on;
end

legend('Pre-LEV DCM', 'Post-LEV DCM', 'Location', 'northwest');

%% Plot all GA params. Supplementary 4
n=1;
for sim = 1:1000
    RMSE_MOGA_LEV_pre(1,sim) = sqrt(mean((data_psd_pre_LEV(n:57) - psd_m_pre_all_LEV(n:57,sim)).^2)); 
    RMSE_MOGA_LEV_post(1,sim) = sqrt(mean((data_psd_post_LEV(n:57) - psd_m_post_all_LEV(n:57,sim)).^2));
    RMSE_MOGA_LTG_pre(1,sim) = sqrt(mean((data_psd_pre_LTG(n:57) - psd_m_pre_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_LTG_post(1,sim) = sqrt(mean((data_psd_post_LTG(n:57) - psd_m_post_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(n:57) - psd_m_pre_all_PL(n:57,sim)).^2));
    RMSE_MOGA_PL_post(1,sim) = sqrt(mean((data_psd_post_PL(n:57) - psd_m_post_all_PL(n:57,sim)).^2));
end
similar_sim_numbers = cell(1, 6);
[RMSE_LEV_pre, indices_LEV_pre] = mink(RMSE_MOGA_LEV_pre, 500); % mink(A)mink(A,k,1) computes the k smallest values in each column of A and
[RMSE_LEV_post, indices_LEV_post] = mink(RMSE_MOGA_LEV_post, 500);
[RMSE_LTG_pre, indices_LTG_pre] = mink(RMSE_MOGA_LTG_pre, 500);
[RMSE_LTG_post, indices_LTG_post] = mink(RMSE_MOGA_LTG_post, 500);
[RMSE_PL_pre, indices_PL_pre] = mink(RMSE_MOGA_PL_pre, 500);
[RMSE_PL_post, indices_PL_post] = mink(RMSE_MOGA_PL_post, 500);
similar_sim_numbers{1} = indices_LEV_pre;
similar_sim_numbers{2} = indices_LEV_post;
similar_sim_numbers{3} = indices_LTG_pre;
similar_sim_numbers{4} = indices_LTG_post;
similar_sim_numbers{5} = indices_PL_pre;
similar_sim_numbers{6} = indices_PL_post;
figure;
for paramIdx = 1:15
    bar_width=linspace(lb(paramIdx),ub(paramIdx),20);
    subplot(3,5, paramIdx);
    bar_width_2=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params = reshape(all_sim_params_pre_PL(paramIdx, similar_sim_numbers{5}), [], 1); 
    post_params = reshape(all_sim_params_post_PL(paramIdx, similar_sim_numbers{6}), [], 1);    
    h_pre = histogram(pre_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'blue', 'EdgeAlpha', 0.8);
    hold on;
    h_post = histogram(post_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'red', 'EdgeAlpha', 0.8);
    hold on;
    [kk,x]=ksdensity(pre_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2,x2]=ksdensity(post_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind = max(h_pre.Values);
    kk = kk/max(kk);
    kk = kk*ind;
    ind2 = max(h_post.Values);
    kk2 = kk2/max(kk2);
    kk2 = kk2*ind2;
    plot(x,kk,'blue');
    hold on;
    plot(x2,kk2,'red');
    hold on;  
    title(paramsvec{paramIdx});
    xlabel('Parameter Value');
    ylabel('Density');
    xlim([lb(paramIdx), ub(paramIdx)]);
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0;
    hold on;
end
legend('Pre-PL', 'Post-PL');

figure;
positions = 1:15;
for Idx = 1:length(positions)
    paramIdx = positions(Idx); % Retrieve the correct index (from positions array)
    subplot(3,5, Idx);
    bar_width=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params = reshape(all_sim_params_pre_LTG(paramIdx, similar_sim_numbers{3}), [], 1); 
    post_params = reshape(all_sim_params_post_LTG(paramIdx, similar_sim_numbers{4}), [], 1);    
    h_pre = histogram(pre_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'blue', 'EdgeAlpha', 0.8);
    hold on;
    h_post = histogram(post_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'red', 'EdgeAlpha', 0.8);
    hold on;
    [kk,x]=ksdensity(pre_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2,x2]=ksdensity(post_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind = max(h_pre.Values);
    kk = kk/max(kk);
    kk = kk*ind;
    ind2 = max(h_post.Values);
    kk2 = kk2/max(kk2);
    kk2 = kk2*ind2;
    plot(x,kk,'blue');
    hold on;
    plot(x2,kk2,'red');
    hold on;
    title(paramsvec{paramIdx});
    xlabel('Parameter Value');
    ylabel('Density');
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0;
    xlim([lb(paramIdx), ub(paramIdx)]);
    hold on;
end
legend('Pre-LTG', 'Post-LTG');
figure;
hold on;
positions = 1:15;
for Idx = 1:length(positions)
    paramIdx = positions(Idx);
    bar_width=linspace(lb(paramIdx),ub(paramIdx),20);
    subplot(3, 5, Idx);
    bar_width_2=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params = reshape(all_sim_params_pre_LEV(paramIdx, similar_sim_numbers{1}), [], 1); 
    post_params = reshape(all_sim_params_post_LEV(paramIdx, similar_sim_numbers{2}), [], 1);    
    h_pre = histogram(pre_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'blue', 'EdgeAlpha', 0.8);
    hold on;
    h_post = histogram(post_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'red', 'EdgeAlpha', 0.8);
    hold on;
    [kk,x]=ksdensity(pre_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2,x2]=ksdensity(post_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind = max(h_pre.Values);
    kk = kk/max(kk);
    kk = kk*ind;
    ind2 = max(h_post.Values);
    kk2 = kk2/max(kk2);
    kk2 = kk2*ind2;
    plot(x,kk,'blue');
    hold on;
    plot(x2,kk2,'red');
    hold on;
    title(paramsvec{paramIdx});
    xlabel('Parameter Value');
    ylabel('Density');
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0;
    xlim([lb(paramIdx), ub(paramIdx)]);
    hold on;
end
legend('Pre-LEV', 'Post-LEV');

%% Plot all DIP-DCM parameters. Supplementary 4
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Hybrid'
load('params_PL_hybrid.mat');
load('log_params_PL_hybrid_posteriors.mat');
load('params_LTG_hybrid.mat');
load('log_params_LTG_hybrid_posteriors.mat');
load('params_LEV_hybrid.mat');
load('log_params_LEV_hybrid_posteriors.mat');
figure;
positions = 1:15;
for Idx = 1:length(positions)
ii = positions(Idx);
%paramIdx = positions(Idx);
muvec_pre=params_pre_DCM_PL(ii,:);
varvec_pre=cov_pre_DCM_PL(ii, :);
muvec_post=params_post_DCM_PL(ii,:);
varvec_post=cov_post_DCM_PL(ii, :);
x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
x_post = linspace(exp(min(muvec_post) - 1.5*sqrt(max(varvec_post))), exp(max(muvec_post) + 1.5*sqrt(max(varvec_post))), 1000);
subplot(3,5,Idx);
hold on;
total_density_pre = zeros(size(x));
total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i)); % Standard deviation is the square root of variance
    pdf_post = (1./(x_post.*sigma_post * sqrt(2 * pi))) .* exp(-(log(x_post) - mu_post).^2 / (2*sigma_post^2));% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/500);
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
plot(x_post, total_density_post, 'r', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
hold off;
end
legend('Pre-PL', 'Post-PL');

figure;
hold on;
positions = 1:15;
for Idx = 1:length(positions)
ii = positions(Idx);
muvec_pre=params_pre_DCM_LTG(ii,:);
varvec_pre=cov_pre_DCM_LTG(ii, :);
muvec_post=params_post_DCM_LTG(ii,:);
varvec_post=cov_post_DCM_LTG(ii, :);
x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
x_post = linspace(exp(min(muvec_post) - 1.5*sqrt(max(varvec_post))), exp(max(muvec_post) + 1.5*sqrt(max(varvec_post))), 1000);
subplot(3,5,Idx);
hold on;
total_density_pre = zeros(size(x));
total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i)); % Standard deviation is the square root of variance
    pdf_post = (1./(x_post.*sigma_post * sqrt(2 * pi))) .* exp(-(log(x_post) - mu_post).^2 / (2*sigma_post^2));% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/500);
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
plot(x_post, total_density_post, 'r', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
hold off;
end
legend('Pre-LTG', 'Post-LTG');

figure;
hold on;
positions = 1:15;
for Idx = 1:length(positions)
ii = positions(Idx);
muvec_pre=params_pre_DCM_LEV(ii,:);
varvec_pre=cov_pre_DCM_LEV(ii, :);
muvec_post=params_post_DCM_LEV(ii,:);
varvec_post=cov_post_DCM_LEV(ii, :);
x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
x_post = linspace(exp(min(muvec_post) - 1.5*sqrt(max(varvec_post))), exp(max(muvec_post) + 1.5*sqrt(max(varvec_post))), 1000);
subplot(3,5,Idx);
hold on;
total_density_pre = zeros(size(x));
total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i)); % Standard deviation is the square root of variance
    pdf_post = (1./(x_post.*sigma_post * sqrt(2 * pi))) .* exp(-(log(x_post) - mu_post).^2 / (2*sigma_post^2));% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/500);
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
plot(x_post, total_density_post, 'r', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
hold off;
end
legend('Pre-LEV', 'Post-LEV');

%% Exmaple distributions. Figure 4B

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_LTG_params.mat'); % 'params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'params_post_DCM_LTG', 'cov_post_DCM_LTG');

figure;
hold on;
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
hold on;
positions = 3; % Define the positions for the parameters
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(2,3, idx);
    log_mean = params_pre_DCM_LTG(i, 1); 
    log_variance = cov_pre_DCM_LTG(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    hold on;
    log_mean_post = params_post_DCM_LTG(i, 1); 
    log_variance_post = cov_post_DCM_LTG(i, 1);
    log_sigma_post = sqrt(log_variance_post);
    x_post = linspace(exp(log_mean_post - 4 * log_sigma_post), exp(log_mean_post + 4 * log_sigma_post), 100);
    y_post = lognpdf(x_post, log_mean_post, log_sigma_post);
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'red'); 
    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    xlim([0.00039, 0.0165]);
    hold on;
end
legend('Pre-LTG DCM', 'Post-LTG DCM', 'Location', 'northwest');


hold on;

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_LTG_params_final.mat'); % 'all_sim_params_pre_LTG', 'all_sim_params_post_LTG');

positions = 3;
for Idx = 1:length(positions)
    paramIdx = positions(Idx); % Retrieve the correct index (from positions array)
    subplot(2,3, Idx+1);
    bar_width=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params = reshape(all_sim_params_pre_LTG(paramIdx, similar_sim_numbers{3}), [], 1); 
    post_params = reshape(all_sim_params_post_LTG(paramIdx, similar_sim_numbers{4}), [], 1);    
    h_pre = histogram(pre_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'blue', 'EdgeAlpha', 0.8);
    hold on;
    h_post = histogram(post_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'red', 'EdgeAlpha', 0.8);
    hold on;
    [kk,x]=ksdensity(pre_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2,x2]=ksdensity(post_params,bar_width_2,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind = max(h_pre.Values);
    kk = kk/max(kk);
    kk = kk*ind;
    ind2 = max(h_post.Values);
    kk2 = kk2/max(kk2);
    kk2 = kk2*ind2;
    %plot(x,kk,'blue');
    hold on;
    %plot(x2,kk2,'red');
    hold on;
    title(paramsvec{paramIdx});
    xlabel('Parameter Value');
    ylabel('Density');
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    xlim([0.00039, 0.0165]);
    hold on;
end
legend('Pre-LTG', 'Post-LTG');


cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Hybrid'
load('params_LTG_hybrid.mat');
load('log_params_LTG_hybrid_posteriors.mat');
hold on;
positions = 3;
for Idx = 1:length(positions)
ii = positions(Idx);
muvec_pre=params_pre_DCM_LTG(ii,:);
varvec_pre=cov_pre_DCM_LTG(ii, :);
muvec_post=params_post_DCM_LTG(ii,:);
varvec_post=cov_post_DCM_LTG(ii, :);
x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
x_post = linspace(exp(min(muvec_post) - 1.5*sqrt(max(varvec_post))), exp(max(muvec_post) + 1.5*sqrt(max(varvec_post))), 1000);
subplot(2,3,Idx+2);
hold on;
total_density_pre = zeros(size(x));
total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i)); % Standard deviation is the square root of variance
    pdf_post = (1./(x_post.*sigma_post * sqrt(2 * pi))) .* exp(-(log(x_post) - mu_post).^2 / (2*sigma_post^2));% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/500);
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
plot(x_post, total_density_post, 'r', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
xlim([0.00039, 0.0165]);
ax = gca; % Get current axis
ax.XAxis.Exponent = 0;
hold off;
end
legend('Pre-LTG', 'Post-LTG');

%% Exmaple distributions. Figure 4A

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_params.mat'); % 'params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'params_post_DCM_LTG', 'cov_post_DCM_LTG');

figure;
hold on;
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
hold on;
positions = 4; % Define the positions for the parameters
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(2,3, idx);
    log_mean = params_pre_DCM_PL(i, 1); 
    log_variance = cov_pre_DCM_PL(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    xlabel('Parameter value (s)');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    xlim([0, 0.05]);
    hold on;
    ax=gca;
    ax.FontSize=16;
end
legend('Pre-PL');


hold on;

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_params_final.mat'); % 'all_sim_params_pre_LTG', 'all_sim_params_post_LTG');

positions = 4;
for Idx = 1:length(positions)
    paramIdx = positions(Idx); % Retrieve the correct index (from positions array)
    subplot(2,3, Idx+1);
    bar_width=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params = reshape(all_sim_params_pre_PL(paramIdx, similar_sim_numbers{3}), [], 1); 
    h_pre = histogram(pre_params,bar_width, 'Normalization', 'pdf', 'FaceColor', 'blue', 'EdgeAlpha', 0.8);
    hold on;
    hold on;
    title(paramsvec{paramIdx});
    xlabel('Parameter value (s)');
    ylabel('Density');
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    xlim([0, 0.05]);
    hold on;
    ax=gca;
    ax.FontSize=16;
end
legend('Pre-PL');


cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Hybrid'
load('params_PL_hybrid.mat');
load('log_params_PL_hybrid_posteriors.mat');
hold on;
positions = 4;
for Idx = 1:length(positions)
ii = positions(Idx);
muvec_pre=params_pre_DCM_PL(ii,:);
varvec_pre=cov_pre_DCM_PL(ii, :);
x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
subplot(2,3,Idx+2);
hold on;
total_density_pre = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
xlabel('Parameter value (s)');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
    xlim([0, 0.05]);
ax = gca; % Get current axis
ax.XAxis.Exponent = 0;
hold off;
    ax=gca;
    ax.FontSize=16;
end
legend('Pre-PL');






