%% Figure 8 efficiency
% raw data from these folders was not provided due to size limits, but the
% code used is below

clearvars
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/spm12/spm12'); % needs spm path before running
addpath('/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency')
%%
% MOGA RMSE
n_values = 500;
for n = n_values
% data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/' ...
%                    'Hybrid_PL_efficiency/' num2str(n) '_gen'];
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen'];
cd(data_folder); 
load('grand_PL_spectrum.mat'); %'average_across_subjects', from data_analysis.m
grand_log_norm_pre_PL_csd = log(movmean(average_across_subjects,5)');
data_psd_pre_PL = grand_log_norm_pre_PL_csd(1:57);
freq_bins = 2:0.5:45;
psd_m_pre_all = zeros(57, 1000);
M = struct(...
    'dipfit', struct('location', '0', 'symmetry', '0', 'modality', 'LFP', 'type', 'LFP', 'symm', '0', 'Lpos', '0', 'Nm', '1', 'Nc', '1', 'model', 'LFP'),...  
    'IS', 'spm_csd_mtf',...
    'g', 'spm_gx_erp',...
    'f', 'spm_fx_lfp',...
    'x', zeros(1, 13),...
    'n', 13,...
    'pE', struct(),...
    'pC', struct(),...
    'hE', 8,...
    'hC', 1/128,...
    'm', 1,...
    'u', 0,...
    'U', 1.0000,...
    'l', 1,...
    'Hz', (2:0.5:30)',...
    'dt', 1);
for sim_pre = 1:1000
    load(['MOGA_LFP_pre_placebo_' num2str(sim_pre) '.mat']);
    x0 = zeros(1,13);
    U=0;
    p = out{1};
    s = out{2}; 
    distances = sqrt(sum(s.^2, 2));
    [~, min_idx] = min(distances);
    param_values = p(min_idx, :);
M.pE = struct(...
    'R', [param_values(1), param_values(2)],...
    'T', [param_values(3), param_values(4)],...
    'G', param_values(5),...
    'H', [param_values(6) param_values(7)  param_values(8) param_values(9) param_values(10)],...
    'A', [param_values(11), param_values(12), param_values(13)],...
    'C', 0,...
    'D', param_values(14),...
    'I', param_values(15),...
    'Lpos', [0; 0; 0],...
    'L', 1,...
    'J', [0 0 0 0 0 0 0 0 1 0 0 0 0],...
   'a', [param_values(16); param_values(17)],...
    'b', [param_values(18); param_values(19)],...
    'c', [param_values(20); param_values(21)],...
    'd', [param_values(22); param_values(23); param_values(24); param_values(25)],...
    'f', [param_values(26); param_values(27)]);
    P=M.pE(1);   
    P.A = num2cell(P.A);
    spectrum_pre=spm_csd_mtf(P,M,U);
    psd_m_pre_current = real(spectrum_pre{1}); 
    psd_m_pre_all_PL(:, sim_pre) = psd_m_pre_current;
end
model_pre_PL_all_500_MOGA=psd_m_pre_all_PL;
RMSE_MOGA_PL_pre=zeros(1,1000);
for sim = 1:1000
    RMSE_MOGA_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(10:57) - psd_m_pre_all_PL(10:57,sim)).^2));
end
[~, indices_PL_pre] = mink(RMSE_MOGA_PL_pre, 500);
similar_sim_numbers = indices_PL_pre;
end
RMSE_MOGA_PL_pre_500=RMSE_MOGA_PL_pre(:,similar_sim_numbers');



%% Plot final fitness against total computation time
% Initialize total runtime arrays
total_runtime_pre_PL_0 = [];
total_runtime_pre_PL_2 = [];
total_runtime_pre_PL_5 = [];
total_runtime_pre_PL_10 = [];
total_runtime_pre_PL_20 = []; 
total_runtime_pre_PL_50 = [];
total_runtime_pre_PL_100 = [];
total_runtime_pre_PL_150 = [];
total_runtime_pre_PL_200 = [];
total_runtime_pre_PL_250 = [];
total_runtime_pre_PL_300 = [];
total_runtime_pre_PL_500 = [];

n_values = [2,5,10,20, 50, 100, 150, 200, 250, 300, 500];
for n = n_values
    data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen'];
    cd(data_folder);
    load(['pre_PL_MOGA_prior_means_500_' num2str(n) '_gen.mat']); % Load necessary variables
    runtime_current_n = zeros(1, 500); % 1x500 vector, 500 repeats

    for idx = 1:length(similar_sim_numbers)
        sim = similar_sim_numbers(idx); 
        load(['MOGA_LFP_pre_placebo_' num2str(sim) '.mat']); 
        if length(out) >= 7
            runtime_current_n(idx) = out{7}; % Runtime value in out
        else
            disp(['Warning: Expected output not found for simulation ' num2str(sim)]);
        end
    end
    %runtime_current_n = runtime_current_n(runtime_current_n ~= 0); 
    eval(['runtime_pre_PL_' num2str(n) ' = runtime_current_n;']);

    data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen/After_DCM'];
    cd(data_folder);
    dcm_runtime_current_n = zeros(1, 500);
    for nsim = 1:500
        load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_31-Oct-2024.mat']);
        dcm_runtime_current_n(nsim) = DCM.runtime_dcm; 
    end
    eval(['dcm_runtime_pre_PL_' num2str(n) ' = dcm_runtime_current_n;']);
    eval(['total_runtime_pre_PL_' num2str(n) ' = runtime_pre_PL_' num2str(n) ' + dcm_runtime_pre_PL_' num2str(n) ';']);
end

% For zero gen there is no MOGA
for n = 0   
    data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen'];
    cd(data_folder);
    dcm_runtime_current_n = zeros(1, 500);
    for nsim = 1:500
        load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_13-Nov-2024.mat']);
        dcm_runtime_current_n(nsim) = DCM.runtime_dcm; 
    end
    eval(['total_runtime_pre_PL_' num2str(n) ' = dcm_runtime_current_n;']);
end


%%
% RMSE
n_values = [2,5,10,20, 50, 100, 150, 200, 250, 300, 500];
for n = n_values
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen/After_DCM'];
cd(data_folder);
model_pre_PL_all=zeros(57,500);
for nsim = 1:500   
    load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_31-Oct-2024.mat']);
    model_pre_PL=spm_csd_mtf(DCM.Ep, DCM.M, DCM.xU);
    model_pre_PL_all(:,nsim)=real(model_pre_PL{1});
    RMSE_hybrid_PL_pre(1,nsim) = sqrt(mean((data_psd_pre_PL(10:57) - model_pre_PL_all(10:57,nsim)).^2));
end
eval(['RMSE_hybrid_PL_pre_' num2str(n) ' = RMSE_hybrid_PL_pre;']);
end

n_values = 0;
for n = n_values
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen'];
cd(data_folder);
model_pre_PL_all=zeros(57,500);
for nsim = 1:500   
    load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_13-Nov-2024.mat']);
    model_pre_PL=spm_csd_mtf(DCM.Ep, DCM.M, DCM.xU);
    model_pre_PL_all(:,nsim)=real(model_pre_PL{1});
    RMSE_hybrid_PL_pre(1,nsim) = sqrt(mean((data_psd_pre_PL(10:57) - model_pre_PL_all(10:57,nsim)).^2));
end
eval(['RMSE_hybrid_PL_pre_' num2str(n) ' = RMSE_hybrid_PL_pre;']);
end


%% Figure 5A, and Supplementary Figure S2A
n_values = [500, 300, 250,200,150,100,50,20,0];
n_values = [0,20,50,100,150,200,250,300,500];
n_values = [0,500];


originalCmap = [
    0.4, 0.7, 1.0;  % Sky Blue
    0.9, 0.7, 0.5;  % Light Brown/Peach
    0.9, 0.9, 0.9;  % Light Grey   
];
numColors = 2;
x = linspace(1, numColors, size(originalCmap, 1));
xi = linspace(1, numColors, numColors);
colors = [interp1(x, originalCmap(:,1), xi, 'linear')', ...
              interp1(x, originalCmap(:,2), xi, 'linear')', ...
              interp1(x, originalCmap(:,3), xi, 'linear')'];

figure;
hold on;

for i = 1:length(n_values)
    n = n_values(i);
    x_data = eval(['total_runtime_pre_PL_' num2str(n)]);
    y_data = eval(['RMSE_hybrid_PL_pre_' num2str(n)]);
    x_mean = mean(x_data, 2);
    y_mean = mean(y_data, 2);
    num_observations = size(x_data, 2);
    x_sem = std(x_data, 0, 2) / sqrt(num_observations);  % SEM for x_data
    y_sem = std(y_data, 0, 2) / sqrt(num_observations);  % SEM for y_data
    errorbar(x_mean, y_mean, y_sem, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10, 'HandleVisibility','off');
    errorbar(x_mean, y_mean, x_sem, 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10, 'HandleVisibility','off');
    plot(x_mean, y_mean, 'o', 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'k', 'MarkerSize', 5); % Mean dot with black edge
end
xlabel('Runtime', 'FontSize', 14);
ylabel('RMSE', 'FontSize', 14);

hold on;

originalCmap = [
    0.2, 0.2, 0.6   % Dark Blue
    0.4, 0.7, 1.0;  % Sky Blue
    0.9, 0.7, 0.5;  % Light Brown/Peach
    0.9, 0.9, 0.9;  % Light Grey   
];
numColors = 10;
x = linspace(1, numColors, size(originalCmap, 1));
xi = linspace(1, numColors, numColors);
colors = [interp1(x, originalCmap(:,1), xi, 'linear')', ...
              interp1(x, originalCmap(:,2), xi, 'linear')', ...
              interp1(x, originalCmap(:,3), xi, 'linear')'];
n_values = 150;
n = n_values;
x_data = eval(['total_runtime_pre_PL_' num2str(n)]);
y_data = eval(['RMSE_hybrid_PL_pre_' num2str(n)]);
x_mean = mean(x_data, 2);
y_mean = mean(y_data, 2);
num_observations = size(x_data, 2);
x_sem = std(x_data, 0, 2) / sqrt(num_observations);  % SEM for x_data
y_sem = std(y_data, 0, 2) / sqrt(num_observations);  % SEM for y_data
errorbar(x_mean, y_mean, y_sem, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10, 'HandleVisibility','off');
errorbar(x_mean, y_mean, x_sem, 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10, 'HandleVisibility','off');
plot(x_mean, y_mean, 'o', 'MarkerFaceColor', colors(1, :), 'MarkerEdgeColor', 'k', 'MarkerSize', 5); % Mean dot with black edge

% Also add MOGA 500 sims
hold on;
x_data = runtime_pre_PL_500;
y_data = RMSE_MOGA_PL_pre_500;
x_mean = mean(x_data, 2);
y_mean = mean(y_data, 2);
num_observations = size(x_data, 2);
x_sem = std(x_data, 0, 2) / sqrt(num_observations);  % SEM for x_data
y_sem = std(y_data, 0, 2) / sqrt(num_observations);  % SEM for y_data
errorbar(x_mean, y_mean, y_sem, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10,'HandleVisibility','off');
errorbar(x_mean, y_mean, x_sem, 'horizontal', 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 2, 'CapSize', 10,'HandleVisibility','off');
hold on;
plot(x_mean, y_mean, 'o', 'MarkerFaceColor', [0.5, 0, 0.125], 'MarkerEdgeColor', [0.5, 0, 0.125], 'MarkerSize', 5,'HandleVisibility','off'); % Mean dot with black edge
yline(mean(RMSE_MOGA_PL_pre_500,2), '--', 'Color', [0.5, 0, 0.125], 'linewidth',1);
%ylim([0, max(y_mean) * 2.5]); 

hold off;
legend('LH', 'DIP, 500', 'DIP, 150', 'GA');



%% Figure 5B
% Plot zero generations hybrid vs data, 150 generations hybrid, and 500 generations MOGA
n_values = [0];
for n = n_values
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen'];
cd(data_folder);
model_pre_PL_all=zeros(57,500);
for nsim = 1:500   
    load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_13-Nov-2024.mat']);
    model_pre_PL=spm_csd_mtf(DCM.Ep, DCM.M, DCM.xU);
    model_pre_PL_all_0(:,nsim)=real(model_pre_PL{1});
end
end
n_values = [150];
for n = n_values
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen/After_DCM'];
cd(data_folder);
model_pre_PL_all=zeros(57,500);
for nsim = 1:500   
    load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_31-Oct-2024.mat']);
    model_pre_PL=spm_csd_mtf(DCM.Ep, DCM.M, DCM.xU);
    model_pre_PL_all_150(:,nsim)=real(model_pre_PL{1});
end
end

n_values = [500];
for n = n_values
data_folder = ['/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_efficiency/Hybrid_PL_efficiency/' num2str(n) '_gen/After_DCM'];
cd(data_folder);
model_pre_PL_all=zeros(57,500);
for nsim = 1:500   
    load(['Grand_pre-PL_LFP_MOGA_means_' num2str(nsim) '_DCM_31-Oct-2024.mat']);
    model_pre_PL=spm_csd_mtf(DCM.Ep, DCM.M, DCM.xU);
    model_pre_PL_all_500(:,nsim)=real(model_pre_PL{1});
end
end

colors = jet(10); 
desaturation_factor = 0.5; 
neutral_color = [0.5, 0.5, 0.5]; 
colors = (1 - desaturation_factor) * colors + desaturation_factor * neutral_color;

figure;
% First plot: 0 gen Hybrid
subplot(2,2,1);
plot(freq_bins(1:57), model_pre_PL_all_0,'Color', colors(4,:), 'LineWidth',0.5, 'HandleVisibility','off');
hold on;
plot(freq_bins(1:57), model_pre_PL_all_0(:,1),'Color', colors(4,:), 'LineWidth',1);
hold on;
mean_0 = mean(model_pre_PL_all_0, 2);
stderr_0 = std(model_pre_PL_all_0, 0, 2) / sqrt(size(model_pre_PL_all_0, 2));
ci_0_upper = mean_0 + 1.96 * stderr_0;
ci_0_lower = mean_0 - 1.96 * stderr_0;
fill([freq_bins(1:57)'; flipud(freq_bins(1:57)')], [ci_0_upper; flipud(ci_0_lower)], colors(9,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility','off');
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--k', 'LineWidth', 1);
ylabel('Log PSD');
xlabel('Frequency (Hz)');
hold on;
legend('LH','Data');
ax = gca;  
ax.FontSize = 14;

subplot(2,2,2);
%plot(freq_bins(1:57), mean_0, 'Color', 'b', 'LineWidth', 1);
ylabel('Log PSD');
xlabel('Frequency (Hz)');
hold on;
plot(freq_bins(1:57), model_pre_PL_all_500, 'Color', colors(9,:), 'LineWidth', 1, 'handlevisibility', 'off');
hold on;
plot(freq_bins(1:57), model_pre_PL_all_500(:,1), 'Color', colors(9,:), 'LineWidth', 1);
hold on;
plot(freq_bins(1:57), data_psd_pre_PL, '--k', 'LineWidth', 1);
%plot(freq_bins(1:57), data_psd_pre_PL, '--k', 'LineWidth', 1);
legend('DIP', 'Data');
ax = gca;  
ax.FontSize = 14;







