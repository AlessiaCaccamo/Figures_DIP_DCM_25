%% Figures - spectra

clearvars
% set path
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25'
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/spm12/spm12'); % needs spm path before running
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25'); %path containing violin plot
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Spectra'); %path containing violin plot

%% Load empirical data
freq_bins = 2:0.5:45;
load('grand_PL_spectrum.mat'); %"average_across_subjects_pre_PL","average_across_subjects_post_PL");
load('grand_LTG_spectrum.mat'); %"average_across_subjects_pre_LTG","average_across_subjects_post_LTG");
load('grand_LEV_spectrum.mat'); %"average_across_subjects_pre_LEV","average_across_subjects_post_LEV");

load('log_PL_spectra.mat'); % data_psd_pre_PL, data_psd_post_PL
load('log_LTG_spectra.mat');
load('log_LEV_spectra.mat');

load('grand_LTG_subject_spectra.mat');
load('grand_LEV_subject_spectra.mat');
load('grand_PL_subject_spectra.mat');

%% Plot spectra, Figure 2

sem_pre_PL=std(all_subject_spectra_pre_PL,1)/sqrt(14);
sem_post_PL=std(all_subject_spectra_post_PL,1)/sqrt(14);
sem_pre_LTG=std(all_subject_spectra_pre_LTG,1)/sqrt(14);
sem_post_LTG=std(all_subject_spectra_post_LTG,1)/sqrt(14);
sem_pre_LEV=std(all_subject_spectra_pre_LEV,1)/sqrt(14);
sem_post_LEV=std(all_subject_spectra_post_LEV,1)/sqrt(14);

figure;
subplot(2,3,1);
plot(freq_bins,average_across_subjects_pre_PL, 'LineWidth',2, 'Color','b');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_pre_PL' - sem_pre_PL'; flip(average_across_subjects_pre_PL' + sem_pre_PL')], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
plot(freq_bins,average_across_subjects_post_PL, 'LineWidth',2, 'Color','r');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_post_PL' - sem_post_PL'; flip(average_across_subjects_post_PL' + sem_post_PL')], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
grid on;
xlabel('Frequency (Hz)', 'FontSize',16);
ylabel('Power Spectral Density', 'FontSize',16);
legend('Pre-Placebo', 'Post-Placebo', 'FontSize',12);
subplot(2,3,2);
plot(freq_bins,average_across_subjects_pre_LTG, 'LineWidth',2, 'Color','b');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_pre_LTG' - sem_pre_LTG'; flip(average_across_subjects_pre_LTG' + sem_pre_LTG')], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
plot(freq_bins,average_across_subjects_post_LTG, 'LineWidth',2, 'Color','r');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_post_LTG' - sem_post_LTG'; flip(average_across_subjects_post_LTG' + sem_post_LTG')], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
grid on;
xlabel('Frequency (Hz)', 'FontSize',16);
ylabel('Power Spectral Density', 'FontSize',16);
legend('Pre-LTG', 'Post-LTG', 'FontSize',12);
subplot(2,3,3);
plot(freq_bins,average_across_subjects_pre_LEV, 'LineWidth',2, 'Color','b');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_pre_LEV' - sem_pre_LEV'; flip(average_across_subjects_pre_LEV' + sem_pre_LEV')], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
plot(freq_bins,average_across_subjects_post_LEV, 'LineWidth',2, 'Color','r');
hold on;
fill([freq_bins'; flip(freq_bins')], [average_across_subjects_post_LEV' - sem_post_LEV'; flip(average_across_subjects_post_LEV' + sem_post_LEV')], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility','off');
grid on;
xlabel('Frequency (Hz)', 'FontSize',16);
ylabel('Power Spectral Density', 'FontSize',16);
legend('Pre-LEV', 'Post-LEV', 'FontSize',12);



%% Load model spectra DCM
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_model_spectra_grand_log_PL_31_May.mat'); % 'model_pre_PL', 'model_post_PL');
load('DCM_LFP_model_spectra_grand_log_LTG_31_May.mat'); % 'model_pre_LTG', 'model_post_LTG');
load('DCM_LFP_model_spectra_grand_log_LEV_31_May.mat'); % 'model_pre_LEV', 'model_post_LEV');

%% %% Load model spectra GA
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_plot_data_models'
load('MOGA_LFP_model_spectra_LEV_grand_150_gen.mat');
psd_m_pre_all_LEV=psd_m_pre_all;
psd_m_post_all_LEV=psd_m_post_all;
load('MOGA_LFP_model_spectra_LTG_grand_150_gen.mat');
psd_m_pre_all_LTG=psd_m_pre_all;
psd_m_post_all_LTG=psd_m_post_all;
load('MOGA_LFP_model_spectra_PL_grand_150_gen.mat');
psd_m_pre_all_PL=psd_m_pre_all;
psd_m_post_all_PL=psd_m_post_all;


%% Replot RMSE violins for hybrid method
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_plot_data_models'
load('models_PL_500_hybrid_150_gen.mat', 'model_pre_PL_all', 'model_post_PL_all');
load('models_LTG_500_hybrid_150_gen.mat', 'model_pre_LTG_all', 'model_post_LTG_all');
load('models_LEV_500_hybrid_150_gen.mat', 'model_pre_LEV_all', 'model_post_LEV_all');

%%
n=10;
for sim = 1:500
    RMSE_hybrid_LEV_pre(1,sim) = sqrt(mean((data_psd_pre_LEV(n:57) - model_pre_LEV_all(n:57,sim)).^2));
    RMSE_hybrid_LEV_post(1,sim) = sqrt(mean((data_psd_post_LEV(n:57) - model_post_LEV_all(n:57,sim)).^2));
    RMSE_hybrid_LTG_pre(1,sim) = sqrt(mean((data_psd_pre_LTG(n:57) - model_pre_LTG_all(n:57,sim)).^2));
    RMSE_hybrid_LTG_post(1,sim) = sqrt(mean((data_psd_post_LTG(n:57) - model_post_LTG_all(n:57,sim)).^2));
    RMSE_hybrid_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(n:57) - model_pre_PL_all(n:57,sim)).^2));
    RMSE_hybrid_PL_post(1,sim) = sqrt(mean((data_psd_post_PL(n:57) - model_post_PL_all(n:57,sim)).^2));
end

%% Figure 3A RMSE for model with 150 generations

navyBlue = [0, 0, 0.5];       
burgundy =  [0.5, 0, 0];       
turquoise = [0.68, 0.8, 0.78];

for sim = 1:1000
    RMSE_MOGA_LEV_pre(1,sim) = sqrt(mean((data_psd_pre_LEV(n:57) - psd_m_pre_all_LEV(n:57,sim)).^2)); 
    RMSE_MOGA_LEV_post(1,sim) = sqrt(mean((data_psd_post_LEV(n:57) - psd_m_post_all_LEV(n:57,sim)).^2));
    RMSE_MOGA_LTG_pre(1,sim) = sqrt(mean((data_psd_pre_LTG(n:57) - psd_m_pre_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_LTG_post(1,sim) = sqrt(mean((data_psd_post_LTG(n:57) - psd_m_post_all_LTG(n:57,sim)).^2));
    RMSE_MOGA_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(n:57) - psd_m_pre_all_PL(n:57,sim)).^2));
    RMSE_MOGA_PL_post(1,sim) = sqrt(mean((data_psd_post_PL(n:57) - psd_m_post_all_PL(n:57,sim)).^2));
end
pre_LEV_spec = real(model_pre_LEV{1});
post_LEV_spec=real(model_post_LEV{1});
pre_LTG_spec=real(model_pre_LTG{1});
post_LTG_spec=real(model_post_LTG{1});
pre_PL_spec=real(model_pre_PL{1});
post_PL_spec=real(model_post_PL{1});

RMSE_DCM_LEV_pre = sqrt(mean((data_psd_pre_LEV(n:57) - pre_LEV_spec(n:57)).^2));
RMSE_DCM_LEV_post = sqrt(mean((data_psd_post_LEV(n:57) - post_LEV_spec(n:57)).^2));
RMSE_DCM_LTG_pre = sqrt(mean((data_psd_pre_LTG(n:57) - pre_LTG_spec(n:57)).^2));
RMSE_DCM_LTG_post = sqrt(mean((data_psd_post_LTG(n:57) - post_LTG_spec(n:57)).^2));
RMSE_DCM_PL_pre = sqrt(mean((data_psd_pre_PL(n:57) - pre_PL_spec(n:57)).^2));
RMSE_DCM_PL_post = sqrt(mean((data_psd_post_PL(n:57) - post_PL_spec(n:57)).^2));


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
%%
RMSE_MLE_pre_PL = RMSE_MOGA_PL_pre(1,similar_sim_numbers{5});
RMSE_MLE_post_PL = RMSE_MOGA_PL_post(1,similar_sim_numbers{6});
RMSE_MLE_pre_LTG = RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3});
RMSE_MLE_post_LTG = RMSE_MOGA_LTG_post(1,similar_sim_numbers{4});
RMSE_MLE_pre_LEV = RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1});
RMSE_MLE_post_LEV = RMSE_MOGA_LEV_post(1,similar_sim_numbers{2});
%save('RMSE_GA_all.mat', 'RMSE_MLE_pre_PL', 'RMSE_MLE_post_PL', 'RMSE_MLE_pre_LTG', 'RMSE_MLE_post_LTG', 'RMSE_MLE_pre_LEV', 'RMSE_MLE_post_LEV');

%% percentages
idx_sup_pre_PL_GA = find(RMSE_MOGA_PL_pre(1,similar_sim_numbers{5})<RMSE_DCM_PL_pre);
idx_inf_pre_PL_GA = find(RMSE_MOGA_PL_pre(1,similar_sim_numbers{5})>RMSE_DCM_PL_pre);
per_sup_pre_PL_GA = length(idx_sup_pre_PL_GA)/length(RMSE_MOGA_PL_pre(1,similar_sim_numbers{5}))*100;
per_inf_pre_PL_GA = length(idx_inf_pre_PL_GA)/length(RMSE_MOGA_PL_pre(1,similar_sim_numbers{5}))*100;

idx_sup_post_PL_GA = find(RMSE_MOGA_PL_post(1,similar_sim_numbers{6})<RMSE_DCM_PL_post);
idx_inf_post_PL_GA = find(RMSE_MOGA_PL_post(1,similar_sim_numbers{6})>RMSE_DCM_PL_post);
per_sup_post_PL_GA = length(idx_sup_post_PL_GA)/length(RMSE_MOGA_PL_post(1,similar_sim_numbers{6}))*100;
per_inf_post_PL_GA = length(idx_inf_post_PL_GA)/length(RMSE_MOGA_PL_post(1,similar_sim_numbers{6}))*100;

idx_sup_pre_LTG_GA = find(RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3})<RMSE_DCM_LTG_pre);
idx_inf_pre_LTG_GA = find(RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3})>RMSE_DCM_LTG_pre);
per_sup_pre_LTG_GA = length(idx_sup_pre_LTG_GA)/length(RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3}))*100;
per_inf_pre_LTG_GA = length(idx_inf_pre_LTG_GA)/length(RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3}))*100;

idx_sup_post_LTG_GA = find(RMSE_MOGA_LTG_post(1,similar_sim_numbers{4})<RMSE_DCM_LTG_post);
idx_inf_post_LTG_GA = find(RMSE_MOGA_LTG_post(1,similar_sim_numbers{4})>RMSE_DCM_LTG_post);
per_sup_post_LTG_GA = length(idx_sup_post_LTG_GA)/length(RMSE_MOGA_LTG_post(1,similar_sim_numbers{4}))*100;
per_inf_post_LTG_GA = length(idx_inf_post_LTG_GA)/length(RMSE_MOGA_LTG_post(1,similar_sim_numbers{4}))*100;

idx_sup_pre_LEV_GA = find(RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1})<RMSE_DCM_LEV_pre);
idx_inf_pre_LEV_GA = find(RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1})>RMSE_DCM_LEV_pre);
per_sup_pre_LEV_GA = length(idx_sup_pre_LEV_GA)/length(RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1}))*100;
per_inf_pre_LEV_GA = length(idx_inf_pre_LEV_GA)/length(RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1}))*100;

idx_sup_post_LEV_GA = find(RMSE_MOGA_LEV_post(1,similar_sim_numbers{2})<RMSE_DCM_LEV_post);
idx_inf_post_LEV_GA = find(RMSE_MOGA_LEV_post(1,similar_sim_numbers{2})>RMSE_DCM_LEV_post);
per_sup_post_LEV_GA = length(idx_sup_post_LEV_GA)/length(RMSE_MOGA_LEV_post(1,similar_sim_numbers{2}))*100;
per_inf_post_LEV_GA = length(idx_inf_post_LEV_GA)/length(RMSE_MOGA_LEV_post(1,similar_sim_numbers{2}))*100;

avg_GA_sup=(per_sup_pre_PL_GA+per_sup_post_PL_GA+per_sup_pre_LTG_GA+per_sup_post_LTG_GA+per_sup_pre_LEV_GA+per_sup_post_LEV_GA)/6;
avg_GA_inf=(per_inf_pre_PL_GA+per_inf_post_PL_GA+per_inf_pre_LTG_GA+per_inf_post_LTG_GA+per_inf_pre_LEV_GA+per_inf_post_LEV_GA)/6;


idx_sup_pre_PL_DIP = find(RMSE_hybrid_PL_pre<RMSE_MOGA_PL_pre(1,similar_sim_numbers{5})); % is done pairwise for the vectors of 500 values, all values decrease even if from the figure it seems like some are similar between GA and DIP, it is because some GA values are ver lage and they still get decreased
idx_inf_pre_PL_DIP = find(RMSE_hybrid_PL_pre>RMSE_MOGA_PL_pre(1,similar_sim_numbers{5}));
per_sup_pre_PL_DIP = length(idx_sup_pre_PL_DIP)/length(RMSE_hybrid_PL_pre)*100;
per_inf_pre_PL_DIP = length(idx_inf_pre_PL_DIP)/length(RMSE_hybrid_PL_pre)*100;

idx_sup_post_PL_DIP = find(RMSE_hybrid_PL_post<RMSE_MOGA_PL_post(1,similar_sim_numbers{6}));
idx_inf_post_PL_DIP = find(RMSE_hybrid_PL_post>RMSE_MOGA_PL_post(1,similar_sim_numbers{6}));
per_sup_post_PL_DIP = length(idx_sup_post_PL_DIP)/length(RMSE_hybrid_PL_post)*100;
per_inf_post_PL_DIP = length(idx_inf_post_PL_DIP)/length(RMSE_hybrid_PL_post)*100;

idx_sup_pre_LTG_DIP = find(RMSE_hybrid_LTG_pre<RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3}));
idx_inf_pre_LTG_DIP = find(RMSE_hybrid_LTG_pre>RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3}));
per_sup_pre_LTG_DIP = length(idx_sup_pre_LTG_DIP)/length(RMSE_hybrid_LTG_pre)*100;
per_inf_pre_LTG_DIP = length(idx_inf_pre_LTG_DIP)/length(RMSE_hybrid_LTG_pre)*100;

idx_sup_post_LTG_DIP = find(RMSE_hybrid_LTG_post<RMSE_MOGA_LTG_post(1,similar_sim_numbers{4}));
idx_inf_post_LTG_DIP = find(RMSE_hybrid_LTG_post>RMSE_MOGA_LTG_post(1,similar_sim_numbers{4}));
per_sup_post_LTG_DIP = length(idx_sup_post_LTG_DIP)/length(RMSE_hybrid_LTG_post)*100;
per_inf_post_LTG_DIP = length(idx_inf_post_LTG_DIP)/length(RMSE_hybrid_LTG_post)*100;

idx_sup_pre_LEV_DIP = find(RMSE_hybrid_LEV_pre<RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1}));
idx_inf_pre_LEV_DIP = find(RMSE_hybrid_LEV_pre>RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1}));
per_sup_pre_LEV_DIP = length(idx_sup_pre_LEV_DIP)/length(RMSE_hybrid_LEV_pre)*100;
per_inf_pre_LEV_DIP = length(idx_inf_pre_LEV_DIP)/length(RMSE_hybrid_LEV_pre)*100;

idx_sup_post_LEV_DIP = find(RMSE_hybrid_LEV_post<RMSE_MOGA_LEV_post(1,similar_sim_numbers{2}));
idx_inf_post_LEV_DIP = find(RMSE_hybrid_LEV_post>RMSE_MOGA_LEV_post(1,similar_sim_numbers{2}));
per_sup_post_LEV_DIP = length(idx_sup_post_LEV_DIP)/length(RMSE_hybrid_LEV_post)*100;
per_inf_post_LEV_DIP = length(idx_inf_post_LEV_DIP)/length(RMSE_hybrid_LEV_post)*100;

avg_DIP_sup=(per_sup_pre_PL_DIP+per_sup_post_PL_DIP+per_sup_pre_LTG_DIP+per_sup_post_LTG_DIP+per_sup_pre_LEV_DIP+per_sup_post_LEV_DIP)/6;
avg_DIP_inf=(per_inf_pre_PL_DIP+per_inf_post_PL_DIP+per_inf_pre_LTG_DIP+per_inf_post_LTG_DIP+per_inf_pre_LEV_DIP+per_inf_post_LEV_DIP)/6;


%%
% Pre-PL
figure;
subplot(3,2,1);
violinPlot1=violinplot(RMSE_MOGA_PL_pre(1,similar_sim_numbers{5}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_PL_pre, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot = violinplot(RMSE_hybrid_PL_pre(:,1:500));
hold on;
scatter(1,RMSE_DCM_PL_pre,90, 'filled', 'markerfacecolor', burgundy);
ylabel('RMSE');
title('Pre-Placebo'); 
set(gca, 'XTick', []);
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.07]);
ax = gca;  
ax.FontSize = 14;

% Post-PL
subplot(3,2,2);
violinPlot1=violinplot(RMSE_MOGA_PL_post(1,similar_sim_numbers{6}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_PL_post, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot=violinplot(RMSE_hybrid_PL_post(:,1:500));
hold on;
scatter(1,RMSE_DCM_PL_post,90, 'filled', 'markerfacecolor',burgundy);
ylabel('RMSE');
title('Post-Placebo');
set(gca, 'XTick', []);
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.07]);
ax = gca;  
ax.FontSize = 14;

% Pre-LTG
subplot(3,2,3);
violinPlot1=violinplot(RMSE_MOGA_LTG_pre(1,similar_sim_numbers{3}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_LTG_pre, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot=violinplot(RMSE_hybrid_LTG_pre(:,1:500));
hold on;
scatter(1,RMSE_DCM_LTG_pre,90, 'filled', 'markerfacecolor', burgundy);
ylabel('RMSE');
title('Pre-LTG');
set(gca, 'XTick', []);
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.085]);
ax = gca;  
ax.FontSize = 14;

% Post-LTG
subplot(3,2,4);
violinPlot1=violinplot(RMSE_MOGA_LTG_post(1,similar_sim_numbers{4}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_LTG_post, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot=violinplot(RMSE_hybrid_LTG_post(:,1:500));
hold on;
scatter(1,RMSE_DCM_LTG_post,90, 'filled', 'markerfacecolor', burgundy);
ylabel('RMSE');
title('Post-LTG');
set(gca, 'XTick', []);
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.085]);
ax = gca;  
ax.FontSize = 14;

% Pre-LEV
subplot(3,2,5);
violinPlot1=violinplot(RMSE_MOGA_LEV_pre(1,similar_sim_numbers{1}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_LEV_pre, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot=violinplot(RMSE_hybrid_LEV_pre(:,1:500));
hold on;
scatter(1,RMSE_DCM_LEV_pre,90,'filled', 'markerfacecolor', burgundy);
ylabel('RMSE');
title('Pre-LEV');
set(gca, 'XTick', []);
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.055]);
ax = gca;  
ax.FontSize = 14;

% Post-LEV
subplot(3,2,6);
violinPlot1=violinplot(RMSE_MOGA_LEV_post(1,similar_sim_numbers{2}));
violinPlot1.ViolinColor = {turquoise};
hold on;
yline(RMSE_DCM_LEV_post, '--', 'LineWidth',1.5, 'Color', burgundy);
hold on;
violinPlot=violinplot(RMSE_hybrid_LEV_post(:,1:500));
hold on;
scatter(1,RMSE_DCM_LEV_post,90,'filled', 'markerfacecolor', burgundy);
ylabel('RMSE'); 
set(gca, 'XTick', []);
title('Post-LEV');
violinPlot.ViolinColor = {navyBlue}; 
ylim([0.02 0.055]);
ax = gca;  
ax.FontSize = 14;

