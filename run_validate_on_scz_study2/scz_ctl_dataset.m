%% Repeat analysis on scz vs control data for validation
%% Load empirical data
clearvars

%cd '/Users/alessiacaccamo/Documents/Exeter/Data/ForAlessia_SZ_VisualGammaVirtualElectrodesSPM/Final_Results_150_gen'
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_validate_on_scz_study2'
load('Spectra_all_subj_norm_final.mat'); % 'average_spectrum', 'average_spectrum_patient', 'f', 'f_patient', 'mas_control', 'mas_patient');
indx=find(f<=85);
mas_control=mas_control(indx);
f=f(indx);
indx_patient=find(f_patient<=85);
mas_patient=mas_patient(indx_patient);
f_patient=f_patient(indx_patient);
data_psd_control=log(mas_control);
data_psd_patient=log(mas_patient);


%% Load model spectra and params Hybrid
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_validate_on_scz_study2'
load('Hybrid_LFP_model_grand_log_ctl_scz_19_Dec.mat'); % 'model_control_all', 'model_patient_all');
load('Hybrid_LFP_params_19_Dec.mat'); % 'params_DCM_ctl', 'params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');
load('Hybrid_LFP_log_params_ctl_scz_19_Dec.mat'); % 'log_params_DCM_ctl', 'log_params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');

%% Load DCM model
load('DCM_LFP_model_grand_log_ctl_scz_19_Dec.mat')


%% Figure 6A

figure;
subplot(1,2,2);
sd_control=std(model_control_all',1);
sd_patient=std(model_patient_all',1);
plot(f,mean(model_control_all,2), 'LineWidth',1.5, 'Color','b');
hold on;
fill([f; flip(f)], [mean(model_control_all,2) - 1.96*(sd_control/sqrt(length(model_control_all)))'; flip(mean(model_control_all,2) + 1.96*(sd_control/sqrt(length(model_control_all)))')], 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
plot(f,data_psd_control,'--', 'LineWidth',1.5, 'Color','b');
hold on;
plot(f,mean(model_patient_all,2), 'LineWidth',1.5, 'Color','r');
hold on;
fill([f; flip(f)], [mean(model_patient_all,2) - 1.96*(sd_patient/sqrt(length(model_patient_all)))'; flip(mean(model_patient_all,2) + 1.96*(sd_patient/sqrt(length(model_patient_all)))')], 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
plot(f,data_psd_patient,'--', 'LineWidth',1.5, 'Color','r');
xlabel('Frequency (Hz)');
ylabel('Log PSD');
legend('Mean Hybrid Ctl', '95% CI', 'Data Ctl', 'Mean Hybrid Scz', '95% CI', 'Data Scz');
xlim([0 90]);
ax = gca;
ax.FontSize = 14;

subplot(1,2,1);
plot(f,model_control, 'LineWidth',1.5, 'Color','b');
hold on;
plot(f,data_psd_control,'--', 'LineWidth',1.5, 'Color','b');
hold on;
plot(f,model_patient, 'LineWidth',1.5, 'Color','r');
hold on;
plot(f,data_psd_patient,'--', 'LineWidth',1.5, 'Color','r');
xlabel('Frequency (Hz)');
ylabel('Log PSD');
legend('DCM Ctl', 'Data Ctl', 'DCM Scz', 'Data Scz');
xlim([0 90]);
ax = gca;
ax.FontSize = 14;


%% Bounds
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
lb(14) = 2*exp(lb_1(14))/1000;
lb(15) = 16*exp(lb_1(15))/1000;
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
ub(14) = 2*exp(ub_1(14))/1000;
ub(15) = 16*exp(ub_1(15))/1000;
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



%% Hybrid
load('Hybrid_LFP_params_19_Dec.mat'); % 'params_DCM_ctl', 'params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');
load('Hybrid_LFP_log_params_ctl_scz_19_Dec.mat'); % 'log_params_DCM_ctl', 'log_params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');

for ii=1:15
muvec_pre=log_params_DCM_ctl(ii,:);
varvec_pre=cov_DCM_ctl(ii, :);
muvec_post=log_params_DCM_scz(ii,:);
varvec_post=cov_DCM_scz(ii, :);
x = linspace(min(muvec_pre) - 3*sqrt(max(varvec_pre)), max(muvec_pre) + 3*sqrt(max(varvec_pre)), 1000); % for log params
total_density_pre = zeros(size(x));
total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i));
    pdf_pre = (1/(sigma_pre * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_pre) / sigma_pre).^2);% pdf of current distribution
    pdf_post = (1/(sigma_post * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_post) / sigma_post).^2);% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/length(muvec_pre)); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    total_density_post = total_density_post + (pdf_post/length(muvec_pre));
end

cumulative_density_pre = cumsum(total_density_pre); % CDF
cumulative_density_pre = cumulative_density_pre / cumulative_density_pre(end); % Normalize CDF
cumulative_density_post = cumsum(total_density_post); % CDF
cumulative_density_post = cumulative_density_post / cumulative_density_post(end); % Normalize CDF

random_samples_pre = zeros(1e5, 1); % Initialize the samples array
random_samples_post = zeros(1e5, 1);
for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre(n) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post(n) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end
%cohen's d
d(ii)=(mean(random_samples_post) - mean(random_samples_pre))/sqrt((var(random_samples_post)+var(random_samples_pre))/2);
end

%save('inferences_hybrid_ctl_scz_150_gen.mat', 'd');



%% Cohen's d DCM
load('DCM_LFP_log_params_ctl_scz_19_Dec.mat')
load('DCM_LFP_params_19_Dec.mat')
d_dcm = zeros(15, 1);

num_samples = 1e5;
for i = 1:15
    mean_ctl= log_params_DCM_ctl(i, 1);
    variance_ctl = cov_DCM_ctl(i, 1);
    sigma_ctl = sqrt(variance_ctl);
    samples_ctl = normrnd(mean_ctl, sigma_ctl, [num_samples, 1]);

    mean_scz = log_params_DCM_scz(i, 1);
    variance_scz = cov_DCM_scz(i, 1);
    sigma_scz = sqrt(variance_scz);
    samples_scz = normrnd(mean_scz, sigma_scz, [num_samples, 1]);

    % cohen's d 
    d_dcm(i) = (mean_scz - mean_ctl)/sqrt((variance_ctl+variance_scz)/2);
end

% save('inferences_dcm_ctl_scz.mat', 'd_dcm');


%% Plot Hybrid and DCM params. Figure 6B

% DCM lognorm posterior distributions
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
load('Hybrid_LFP_params_19_Dec.mat'); % 'params_DCM_ctl', 'params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');
load('Hybrid_LFP_log_params_ctl_scz_19_Dec.mat'); % 'log_params_DCM_ctl', 'log_params_DCM_scz', 'cov_DCM_ctl', 'cov_DCM_scz');

figure;
positions = 3; % Define the positions for the parameters
% for idx = 1:length(positions)
for idx = 1
    ii = positions(idx); % Retrieve the correct index (from positions array)
    subplot(1,2, idx+1);
    %subplot(3,5, idx);
    muvec_pre=params_DCM_ctl(ii,:);
    varvec_pre=cov_DCM_ctl(ii, :);
    muvec_post=params_DCM_scz(ii,:);
    varvec_post=cov_DCM_scz(ii, :);
    x = linspace(exp(min(muvec_pre) - 1.5*sqrt(max(varvec_pre))), exp(max(muvec_pre) + 1.5*sqrt(max(varvec_pre))), 1000); % for log params
    x_post = linspace(exp(min(muvec_post) - 1.5*sqrt(max(varvec_post))), exp(max(muvec_post) + 1.5*sqrt(max(varvec_post))), 1000);
    hold on;
    total_density_pre = zeros(size(x));
    total_density_post = zeros(size(x));
for i = 1:length(muvec_pre)
    mu_pre = muvec_pre(i);
    sigma_pre = sqrt(varvec_pre(i)); % Standard deviation is the square root of variance
    pdf_pre = (1./(x.*sigma_pre * sqrt(2 * pi))) .* exp(-(log(x) - mu_pre).^2 / (2*sigma_pre^2));% pdf of current distribution
    total_density_pre = total_density_pre + (pdf_pre/length(muvec_pre)); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    mu_post = muvec_post(i);
    sigma_post = sqrt(varvec_post(i)); % Standard deviation is the square root of variance
    pdf_post = (1./(x_post.*sigma_post * sqrt(2 * pi))) .* exp(-(log(x_post) - mu_post).^2 / (2*sigma_post^2));% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/length(muvec_pre));
end
plot(x, total_density_pre, 'b', 'LineWidth', 2); 
plot(x_post, total_density_post, 'r', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
ax = gca; 
ax.XAxis.Exponent = 0; 
set(ax, 'FontSize', 15, 'Box', 'on');
xlim([0.001 0.005]);
hold off;
end
legend('Ctl', 'Scz');

% Load params DCM
load('DCM_LFP_log_params_ctl_scz_19_Dec.mat')
load('DCM_LFP_params_19_Dec.mat')

%figure;
hold on;
positions = 3 ; % Define the positions for the parameters
% for idx = 1:length(positions)
for idx = 1
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(1,2, idx);
    %subplot(3,5, idx);
    log_mean = params_DCM_ctl(i, 1); 
    log_variance = cov_DCM_ctl(i, 1);
    log_sigma = sqrt(log_variance);
    x = linspace(exp(log_mean - 4 * log_sigma), exp(log_mean + 4 * log_sigma), 100);
    y = lognpdf(x, log_mean, log_sigma);
    plot(x, y, 'LineWidth', 2, 'Color', 'blue'); 
    hold on;
    log_mean_post = params_DCM_scz(i, 1); 
    log_variance_post = cov_DCM_scz(i, 1);
    log_sigma_post = sqrt(log_variance_post);
    x_post = linspace(exp(log_mean_post - 4 * log_sigma_post), exp(log_mean_post + 4 * log_sigma_post), 100);
    y_post = lognpdf(x_post, log_mean_post, log_sigma_post);
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'red'); 
    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; 
    ax.XAxis.Exponent = 0; 
    set(ax, 'FontSize', 15, 'Box', 'on');
   xlim([0.001 0.005]);
    hold on;
end
legend('Ctl DCM', 'Scz DCM', 'Location', 'northwest');



