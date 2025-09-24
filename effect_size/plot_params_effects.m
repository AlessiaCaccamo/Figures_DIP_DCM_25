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



%% Differences between estimates and Bayesian credible intervals
% Standard DCM
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/DCM'
load('DCM_LFP_PL_log_params_final.mat'); % 'log_params_pre_DCM_PL', 'cov_pre_DCM_PL', 'log_params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_log_params_final.mat'); % 'log_params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'log_params_post_DCM_LTG', 'cov_post_DCM_LTG');
% Empirically -> sample. 
%Generates empirical samples from the pre and post distributions.
%Computes the sample-based distribution of differences.
%Extracts the empirical 95% percentile interval, which is the Bayesian credible interval or a non-parametric CI.
for ii = 1:15
    mu_pre = log_params_pre_DCM_PL(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_PL(ii, 1));
    mu_post = log_params_post_DCM_PL(ii, 1);
    sigma_post = sqrt(cov_post_DCM_PL(ii, 1));
    N = 1e5;
    random_samples_pre_PL_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_PL_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
    diff_samples = random_samples_post_PL_DCM(:,ii) - random_samples_pre_PL_DCM(:,ii);
    mu_diff_PL(ii) = mean(diff_samples); % unnormalised effect
    BCI_PL(ii, :) = prctile(diff_samples, [2.5 97.5]);
    % normalised effect, cohen's d
    d_dcm_PL(ii) = (mean(random_samples_post_PL_DCM(:,ii))-mean(random_samples_pre_PL_DCM(:,ii)))/sqrt((var(random_samples_post_PL_DCM(:,ii))+var(random_samples_pre_PL_DCM(:,ii)))/2);
end

for ii = 1:15
    mu_pre = log_params_pre_DCM_LTG(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LTG(ii, 1));
    mu_post = log_params_post_DCM_LTG(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LTG(ii, 1));
    N = 1e5;
    random_samples_pre_LTG_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LTG_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
    diff_samples = random_samples_post_LTG_DCM(:,ii) - random_samples_pre_LTG_DCM(:,ii);
    mu_diff_LTG(ii) = mean(diff_samples);
    BCI_LTG(ii, :) = prctile(diff_samples, [2.5 97.5]);
    %d_dcm_LTG(i) = (mu_post-mu_pre)/sqrt((sigma_post^2+sigma_pre^2)/2); %
    %this and the below give the same result
    d_dcm_LTG(ii) = (mean(random_samples_post_LTG_DCM(:,ii))-mean(random_samples_pre_LTG_DCM(:,ii)))/sqrt((var(random_samples_post_LTG_DCM(:,ii))+var(random_samples_pre_LTG_DCM(:,ii)))/2);
end
% cohen's d gives also the same reesult sampling and using the analytical
% formula
for ii = 1:15
    mu_pre = log_params_pre_DCM_LEV(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LEV(ii, 1));
    mu_post = log_params_post_DCM_LEV(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LEV(ii, 1));
    N = 1e5;
    random_samples_pre_LEV_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LEV_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
    diff_samples = random_samples_post_LEV_DCM(:,ii) - random_samples_pre_LEV_DCM(:,ii);
    mu_diff_LEV(ii) = mean(diff_samples);
    BCI_LEV(ii, :) = prctile(diff_samples, [2.5 97.5]);
    d_dcm_LEV(ii) = (mean(random_samples_post_LEV_DCM(:,ii))-mean(random_samples_pre_LEV_DCM(:,ii)))/sqrt((var(random_samples_post_LEV_DCM(:,ii))+var(random_samples_pre_LEV_DCM(:,ii)))/2);
end


%%
% DIP DCM
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/Hybrid'
load('params_LEV_hybrid.mat')
load('params_PL_hybrid.mat')
load('params_LTG_hybrid.mat')
load('log_params_PL_hybrid_posteriors.mat');
load('log_params_LTG_hybrid_posteriors.mat');
load('log_params_LEV_hybrid_posteriors.mat');

random_samples_pre_PL_DIP = zeros(1e5, 15); % Initialize the samples array
random_samples_post_PL_DIP = zeros(1e5, 15);

random_samples_pre_LTG_DIP = zeros(1e5, 15); % Initialize the samples array
random_samples_post_LTG_DIP = zeros(1e5, 15);

random_samples_pre_LEV_DIP = zeros(1e5, 15); % Initialize the samples array
random_samples_post_LEV_DIP = zeros(1e5, 15);

for ii=1:15
muvec_pre=log_posteriors_pre_DCM_PL(ii,:);
varvec_pre=cov_pre_DCM_PL(ii, :);
muvec_post=log_posteriors_post_DCM_PL(ii,:);
varvec_post=cov_post_DCM_PL(ii, :);
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
    %plot(x, pdf, 'red', 'LineWidth', 0.5, 'HandleVisibility','off'); % Individual distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    total_density_post = total_density_post + (pdf_post/500);
end
cumulative_density_pre = cumsum(total_density_pre); % CDF
cumulative_density_pre = cumulative_density_pre / cumulative_density_pre(end); % Normalize CDF
cumulative_density_post = cumsum(total_density_post); % CDF
cumulative_density_post = cumulative_density_post / cumulative_density_post(end); % Normalize CDF


for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_PL_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_PL_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

diff_samples = random_samples_post_PL_DIP(:,ii) - random_samples_pre_PL_DIP(:,ii);

d_DIP_PL(ii) = mean(diff_samples);
%d_DIP_PL(ii)=mean(random_samples_post) - mean(random_samples_pre);
BCI_2_DIP_PL(ii,:) = prctile(diff_samples, [2.5, 97.5]);

end

for ii=1:15
muvec_pre=log_posteriors_pre_DCM_LTG(ii,:);
varvec_pre=cov_pre_DCM_LTG(ii, :);
muvec_post=log_posteriors_post_DCM_LTG(ii,:);
varvec_post=cov_post_DCM_LTG(ii, :);
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
    %plot(x, pdf, 'red', 'LineWidth', 0.5, 'HandleVisibility','off'); % Individual distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    total_density_post = total_density_post + (pdf_post/500);
end
cumulative_density_pre = cumsum(total_density_pre); % CDF
cumulative_density_pre = cumulative_density_pre / cumulative_density_pre(end); % Normalize CDF
cumulative_density_post = cumsum(total_density_post); % CDF
cumulative_density_post = cumulative_density_post / cumulative_density_post(end); % Normalize CDF

for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LTG_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LTG_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

diff_samples = random_samples_post_LTG_DIP(:,ii) - random_samples_pre_LTG_DIP(:,ii);
d_DIP_LTG(ii) = mean(diff_samples);
%d_DIP_LTG(ii)=mean(random_samples_post) - mean(random_samples_pre);
BCI_2_DIP_LTG(ii, :) = prctile(diff_samples, [2.5, 97.5]);

end

for ii=1:15
muvec_pre=log_posteriors_pre_DCM_LEV(ii,:);
varvec_pre=cov_pre_DCM_LEV(ii, :);
muvec_post=log_posteriors_post_DCM_LEV(ii,:);
varvec_post=cov_post_DCM_LEV(ii, :);
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
    %plot(x, pdf, 'red', 'LineWidth', 0.5, 'HandleVisibility','off'); % Individual distribution
    total_density_pre = total_density_pre + (pdf_pre/500); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    total_density_post = total_density_post + (pdf_post/500);
end
cumulative_density_pre = cumsum(total_density_pre); % CDF
cumulative_density_pre = cumulative_density_pre / cumulative_density_pre(end); % Normalize CDF
cumulative_density_post = cumsum(total_density_post); % CDF
cumulative_density_post = cumulative_density_post / cumulative_density_post(end); % Normalize CDF


for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LEV_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LEV_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

diff_samples = random_samples_post_LEV_DIP(:,ii) - random_samples_pre_LEV_DIP(:,ii);
d_DIP_LEV(ii) = mean(diff_samples);
%d_DIP_LEV(ii)=mean(random_samples_post) - mean(random_samples_pre);
BCI_2_DIP_LEV(ii, :) = prctile(diff_samples, [2.5, 97.5]);
end



%% GA
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/GA'
load('MOGA_LFP_PL_log_params_final.mat'); % 'all_sim_log_params_pre_PL', 'all_sim_log_params_post_PL');
load('MOGA_LFP_LTG_log_params_final.mat'); % 'all_sim_log_params_pre_LTG', 'all_sim_log_params_post_LTG');
load('MOGA_LFP_LEV_log_params_final.mat'); % 'all_sim_log_params_pre_LEV', 'all_sim_log_params_post_LEV');
random_samples_pre_PL_GA = zeros(1e5, 15); 
random_samples_post_PL_GA = zeros(1e5, 15);
random_samples_pre_LTG_GA = zeros(1e5, 15); 
random_samples_post_LTG_GA = zeros(1e5, 15);
random_samples_pre_LEV_GA = zeros(1e5, 15); 
random_samples_post_LEV_GA = zeros(1e5, 15);
lb=lb_1;
ub=ub_1;
for paramIdx = 1:15
    bar_width_PL=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2_PL=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params_PL = reshape(all_sim_log_params_pre_PL(paramIdx, similar_sim_numbers{5}'), [], 1); 
    post_params_PL = reshape(all_sim_log_params_post_PL(paramIdx, similar_sim_numbers{6}'), [], 1);    
    [h_pre_PL,~] = histcounts(pre_params_PL,bar_width_PL, 'Normalization', 'pdf');
    [h_post_PL,~] = histcounts(post_params_PL,bar_width_PL, 'Normalization', 'pdf');
    [kk_PL,x_PL]=ksdensity(pre_params_PL,bar_width_2_PL,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2_PL,x2_PL]=ksdensity(post_params_PL,bar_width_2_PL,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind_PL = max(h_pre_PL);
    kk_PL = kk_PL/max(kk_PL);
    kk_PL = kk_PL*ind_PL;
    ind2_PL = max(h_post_PL);
    kk2_PL = kk2_PL/max(kk2_PL);
    kk2_PL = kk2_PL*ind2_PL;
    
    bar_width_LTG=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2_LTG=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params_LTG = reshape(all_sim_log_params_pre_LTG(paramIdx, similar_sim_numbers{3}'), [], 1); 
    post_params_LTG = reshape(all_sim_log_params_post_LTG(paramIdx, similar_sim_numbers{4}'), [], 1);    
    [h_pre_LTG,~] = histcounts(pre_params_LTG,bar_width_LTG, 'Normalization', 'pdf');
    [h_post_LTG,~] = histcounts(post_params_LTG,bar_width_LTG, 'Normalization', 'pdf');
    [kk_LTG,x_LTG]=ksdensity(pre_params_LTG,bar_width_2_LTG,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2_LTG,x2_LTG]=ksdensity(post_params_LTG,bar_width_2_LTG,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind_LTG = max(h_pre_LTG);
    kk_LTG = kk_LTG/max(kk_LTG);
    kk_LTG = kk_LTG*ind_LTG;
    ind2_LTG = max(h_post_LTG);
    kk2_LTG = kk2_LTG/max(kk2_LTG);
    kk2_LTG = kk2_LTG*ind2_LTG;

    bar_width_LEV=linspace(lb(paramIdx),ub(paramIdx),20);
    bar_width_2_LEV=linspace(lb(paramIdx),ub(paramIdx),1000);
    pre_params_LEV = reshape(all_sim_log_params_pre_LEV(paramIdx, similar_sim_numbers{1}'), [], 1); 
    post_params_LEV = reshape(all_sim_log_params_post_LEV(paramIdx, similar_sim_numbers{2}'), [], 1);    
    [h_pre_LEV,~] = histcounts(pre_params_LEV,bar_width_LEV, 'Normalization', 'pdf');
    [h_post_LEV,~] = histcounts(post_params_LEV,bar_width_LEV, 'Normalization', 'pdf');
    [kk_LEV,x_LEV]=ksdensity(pre_params_LEV,bar_width_2_LEV,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    [kk2_LEV,x2_LEV]=ksdensity(post_params_LEV,bar_width_2_LEV,'Bandwidth',(ub(paramIdx)-lb(paramIdx))*0.03,'Support',[lb(paramIdx),ub(paramIdx)],'BoundaryCorrection','reflection');
    ind_LEV = max(h_pre_LEV);
    kk_LEV = kk_LEV/max(kk_LEV);
    kk_LEV = kk_LEV*ind_LEV;
    ind2_LEV = max(h_post_LEV);
    kk2_LEV = kk2_LEV/max(kk2_LEV);
    kk2_LEV = kk2_LEV*ind2_LEV;    


cdf_pre_PL = cumsum(kk_PL) / sum(kk_PL);  % Normalize to create a CDF
cdf_post_PL = cumsum(kk2_PL) / sum(kk2_PL);
cdf_pre_LTG = cumsum(kk_LTG) / sum(kk_LTG);  
cdf_post_LTG = cumsum(kk2_LTG) / sum(kk2_LTG);
cdf_pre_LEV = cumsum(kk_LEV) / sum(kk_LEV);
cdf_post_LEV = cumsum(kk2_LEV) / sum(kk2_LEV);
for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_PL(n) = x_PL(find(cdf_pre_PL >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_PL(n) = x2_PL(find(cdf_post_PL >= rand_num, 1, 'first'));
    random_samples_pre_LTG(n) = x_LTG(find(cdf_pre_LTG >= rand_num, 1, 'first')); 
    random_samples_post_LTG(n) = x2_LTG(find(cdf_post_LTG >= rand_num, 1, 'first'));
    random_samples_pre_LEV(n) = x_LEV(find(cdf_pre_LEV >= rand_num, 1, 'first')); 
    random_samples_post_LEV(n) = x2_LEV(find(cdf_post_LEV >= rand_num, 1, 'first'));
end

% figure;
% histogram(random_samples_pre_PL, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5); % Histogram of samples
random_samples_pre_PL_GA(:,paramIdx)=random_samples_pre_PL';
random_samples_post_PL_GA(:,paramIdx)=random_samples_post_PL';
random_samples_pre_LTG_GA(:,paramIdx)=random_samples_pre_LTG';
random_samples_post_LTG_GA(:,paramIdx)=random_samples_post_LTG';
random_samples_pre_LEV_GA(:,paramIdx)=random_samples_pre_LEV';
random_samples_post_LEV_GA(:,paramIdx)=random_samples_post_LEV';

diff_samples_PL = random_samples_post_PL_GA(:,paramIdx) - random_samples_pre_PL_GA(:,paramIdx);
d_GA_PL(paramIdx) = mean(diff_samples_PL);
BCI_2_GA_PL(paramIdx, :) = prctile(diff_samples_PL, [2.5, 97.5]);

diff_samples_LTG = random_samples_post_LTG_GA(:,paramIdx) - random_samples_pre_LTG_GA(:,paramIdx);
d_GA_LTG(paramIdx) = mean(diff_samples_LTG);
BCI_2_GA_LTG(paramIdx, :) = prctile(diff_samples_LTG, [2.5, 97.5]);

diff_samples_LEV = random_samples_post_LEV_GA(:,paramIdx) - random_samples_pre_LEV_GA(:,paramIdx);
d_GA_LEV(paramIdx) = mean(diff_samples_LEV); 
BCI_2_GA_LEV(paramIdx, :) = prctile(diff_samples_LEV, [2.5, 97.5]);
end



%% Inferences. Figure 4C
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/effect_size'
load('data_cohensd.mat'); % 'd_dcm_PL', 'd_dcm_LTG', 'd_dcm_LEV', 'd_PL', 'd_LTG', 'd_LEV', 'd_MOGA_PL', 'd_MOGA_LTG', 'd_MOGA_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/effect_size'
load('raw_effects_BCI_DCM.mat', 'mu_diff_PL', 'BCI_PL', 'mu_diff_LTG', 'BCI_LTG', 'mu_diff_LEV', 'BCI_LEV')
load('raw_effects_BCI_DIP.mat', 'd_DIP_PL', 'BCI_2_DIP_PL', 'd_DIP_LTG', 'BCI_2_DIP_LTG', 'd_DIP_LEV', 'BCI_2_DIP_LEV')
load('raw_effects_BCI_GA.mat', 'd_GA_PL', 'BCI_2_GA_PL', 'd_GA_LTG', 'BCI_2_GA_LTG', 'd_GA_LEV', 'BCI_2_GA_LEV')


figure;
% ======= 1. PL =======
subplot(3,3,1);
valid_idx = ...
    (abs(d_dcm_PL) >= 0.2) & ...                      
    ((BCI_PL(:,1) > 0 & BCI_PL(:,2) > 0) | ...       
    (BCI_PL(:,1) < 0 & BCI_PL(:,2) < 0) );          

filtered_mu_diff = mu_diff_PL(valid_idx)';
filtered_BCI = BCI_PL(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_BCI(:,2) - filtered_BCI(:,1));
[~, sorted_indices] = sort(ci_width, 'ascend');

b = bar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
    'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_mu_diff - filtered_BCI(:,1)';
upper_error = filtered_BCI(:,2)' - filtered_mu_diff;
errorbar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
         lower_error(sorted_indices), upper_error(sorted_indices), ...
         'k', 'LineWidth', 1.5, 'LineStyle', 'none');
ylabel('Post-Pre');
title('DCM, PL');
xticks(1:length(sorted_indices));
xticklabels(filtered_params(sorted_indices));
xtickangle(90);
ax = gca; ax.XGrid = 'on'; ax.YGrid = 'off'; ax.FontSize = 14;

% ======= 2. LTG =======
subplot(3,3,2);
valid_idx = ...
    (abs(d_dcm_LTG) >= 0.2) & ...                      
    ((BCI_LTG(:,1) > 0 & BCI_LTG(:,2) > 0) | ...       
    (BCI_LTG(:,1) < 0 & BCI_LTG(:,2) < 0) );   

filtered_mu_diff = mu_diff_LTG(valid_idx)';
filtered_BCI = BCI_LTG(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_BCI(:,2) - filtered_BCI(:,1));
[~, sorted_indices] = sort(ci_width, 'ascend');

b = bar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
    'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_mu_diff - filtered_BCI(:,1)';
upper_error = filtered_BCI(:,2)' - filtered_mu_diff;
errorbar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
         lower_error(sorted_indices), upper_error(sorted_indices), ...
         'k', 'LineWidth', 1.5, 'LineStyle', 'none');
ylabel('Post-Pre');
title('DCM, LTG');
xticks(1:length(sorted_indices));
xticklabels(filtered_params(sorted_indices));
xtickangle(90);
ax = gca; ax.XGrid = 'on'; ax.YGrid = 'off'; ax.FontSize = 14;

% ======= 3. LEV =======
subplot(3,3,3);
valid_idx = ...
    (abs(d_dcm_LEV) >= 0.2) & ...                      
    ((BCI_LEV(:,1) > 0 & BCI_LEV(:,2) > 0) | ...       
    (BCI_LEV(:,1) < 0 & BCI_LEV(:,2) < 0) );   

filtered_mu_diff = mu_diff_LEV(valid_idx)';
filtered_BCI = BCI_LEV(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_BCI(:,2) - filtered_BCI(:,1));
[~, sorted_indices] = sort(ci_width, 'ascend');

b = bar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
    'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_mu_diff - filtered_BCI(:,1)';
upper_error = filtered_BCI(:,2)' - filtered_mu_diff;
errorbar(1:length(sorted_indices), filtered_mu_diff(sorted_indices), ...
         lower_error(sorted_indices), upper_error(sorted_indices), ...
         'k', 'LineWidth', 1.5, 'LineStyle', 'none');
ylabel('Post-Pre');
title('DCM, LEV');
xticks(1:length(sorted_indices));
xticklabels(filtered_params(sorted_indices));
xtickangle(90);
ax = gca; ax.XGrid = 'on'; ax.YGrid = 'off'; ax.FontSize = 14;

% ======= 1. GA PL =======

subplot(3,3,4);

valid_idx = ...
    (abs(d_MOGA_PL') >= 0.2) & ...                      
    ((BCI_2_GA_PL(:,1) > 0 & BCI_2_GA_PL(:,2) > 0) | ...       
    (BCI_2_GA_PL(:,1) < 0 & BCI_2_GA_PL(:,2) < 0) );   

filtered_diff = d_GA_PL(valid_idx);
filtered_CI = BCI_2_GA_PL(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
title('GA, PL');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('GA, PL');

% ======= 2. GA LTG =======
subplot(3,3,5);
valid_idx = ...
    (abs(d_MOGA_LTG') >= 0.2) & ...                      
    ((BCI_2_GA_LTG(:,1) > 0 & BCI_2_GA_LTG(:,2) > 0) | ...       
    (BCI_2_GA_LTG(:,1) < 0 & BCI_2_GA_LTG(:,2) < 0) );   

filtered_diff = d_GA_LTG(valid_idx);
filtered_CI = BCI_2_GA_LTG(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('GA, LTG');

% ======= 3. GA LEV =======
subplot(3,3,6);
valid_idx = ...
    (abs(d_MOGA_LEV') >= 0.2) & ...                      
    ((BCI_2_GA_LEV(:,1) > 0 & BCI_2_GA_LEV(:,2) > 0) | ...       
    (BCI_2_GA_LEV(:,1) < 0 & BCI_2_GA_LEV(:,2) < 0) );   

filtered_diff = d_GA_LEV(valid_idx);
filtered_CI = BCI_2_GA_LEV(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('GA, LEV');


% ======= 1. DIP PL =======

subplot(3,3,7);

valid_idx = ...
    (abs(d_PL') >= 0.2) & ...                      
    ((BCI_2_DIP_PL(:,1) > 0 & BCI_2_DIP_PL(:,2) > 0) | ...       
    (BCI_2_DIP_PL(:,1) < 0 & BCI_2_DIP_PL(:,2) < 0) );   

filtered_diff = d_DIP_PL(valid_idx);
filtered_CI = BCI_2_DIP_PL(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('DIP, PL');

% ======= 2. DIP LTG =======
subplot(3,3,8);
valid_idx = ...
    (abs(d_LTG') >= 0.2) & ...                      
    ((BCI_2_DIP_LTG(:,1) > 0 & BCI_2_DIP_LTG(:,2) > 0) | ...       
    (BCI_2_DIP_LTG(:,1) < 0 & BCI_2_DIP_LTG(:,2) < 0) );   

filtered_diff = d_DIP_LTG(valid_idx);
filtered_CI = BCI_2_DIP_LTG(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);
title('DIP, LTG');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;

% ======= 3. DIP LEV =======
subplot(3,3,9);
valid_idx = ...
    (abs(d_LEV') >= 0.2) & ...                      
    ((BCI_2_DIP_LEV(:,1) > 0 & BCI_2_DIP_LEV(:,2) > 0) | ...       
    (BCI_2_DIP_LEV(:,1) < 0 & BCI_2_DIP_LEV(:,2) < 0) );   

filtered_diff = d_DIP_LEV(valid_idx);
filtered_CI = BCI_2_DIP_LEV(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Post-Pre');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('DIP, LEV');



%% Scz vs Ctl. Inferences Figure 5D
% Standard DCM
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_validate_on_scz/raw_data_and_scripts/Final_Results_150_gen/After_DCM'
load('DCM_LFP_log_params_ctl_scz_19_Dec.mat')
load('DCM_LFP_params_19_Dec.mat')

for ii = 1:15
    mu_pre = log_params_DCM_ctl(ii,1);
    sigma_pre = sqrt(cov_DCM_ctl(ii, 1));
    mu_post = log_params_DCM_scz(ii,1);
    sigma_post = sqrt(cov_DCM_scz(ii, 1));
    N = 1e5;
    random_samples_ctl_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_scz_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
    diff_samples = random_samples_scz_DCM(:,ii) - random_samples_ctl_DCM(:,ii);
    d_DCM(ii) = mean(diff_samples);
    BCI_DCM(ii, :) = prctile(diff_samples, [2.5 97.5]);
end


% DIP
load('Hybrid_LFP_log_params_ctl_scz_19_Dec.mat')
load('Hybrid_LFP_params_19_Dec.mat')
random_samples_ctl_DIP = zeros(1e5, 15); % Initialize the samples array
random_samples_scz_DIP = zeros(1e5, 15);

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
    %plot(x, pdf, 'red', 'LineWidth', 0.5, 'HandleVisibility','off'); % Individual distribution
    total_density_pre = total_density_pre + (pdf_pre/400); % each pdf contributes 1/500 (weighted) to the overall sum, which adds up to 1. 
    total_density_post = total_density_post + (pdf_post/400);
end
cumulative_density_pre = cumsum(total_density_pre); % CDF
cumulative_density_pre = cumulative_density_pre / cumulative_density_pre(end); % Normalize CDF
cumulative_density_post = cumsum(total_density_post); % CDF
cumulative_density_post = cumulative_density_post / cumulative_density_post(end); % Normalize CDF


for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_ctl_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_scz_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

diff_samples = random_samples_scz_DIP(:,ii) - random_samples_ctl_DIP(:,ii);

d_DIP(ii) = mean(diff_samples);
%d_DIP_PL(ii)=mean(random_samples_post) - mean(random_samples_pre);
BCI_DIP(ii,:) = prctile(diff_samples, [2.5, 97.5]);

end

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_validate_on_scz/raw_data_and_scripts'
load('inferences_dcm_ctl_scz_150_gen.mat')

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/effect_size'
% ======= 1. DCM =======
figure;
subplot(3,3,1);

valid_idx = ...
    (abs(d_dcm) >= 0.2) & ...                      
    ((BCI_DCM(:,1) > 0 & BCI_DCM(:,2) > 0) | ...       
    (BCI_DCM(:,1) < 0 & BCI_DCM(:,2) < 0) );   

filtered_diff = d_DCM(valid_idx);
filtered_CI = BCI_DCM(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Scz - Ctl');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);

ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;
title('DCM');

% ======= 2. DIP =======
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_validate_on_scz/raw_data_and_scripts'
load('inferences_hybrid_ctl_scz_150_gen.mat')
subplot(3,3,2);
valid_idx = ...
    (abs(d') >= 0.2) & ...                      
    ((BCI_DIP(:,1) > 0 & BCI_DIP(:,2) > 0) | ...       
    (BCI_DIP(:,1) < 0 & BCI_DIP(:,2) < 0) );   

filtered_diff = d_DIP(valid_idx);
filtered_CI = BCI_DIP(valid_idx, :);
filtered_params = paramsvec(valid_idx);

ci_width = abs(filtered_CI(:,2) - filtered_CI(:,1));
[~, sorted_idx] = sort(ci_width, 'descend');

b = bar(1:length(sorted_idx), filtered_diff(sorted_idx), 'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

lower_error = filtered_diff(sorted_idx) - filtered_CI(sorted_idx,1)';
upper_error = filtered_CI(sorted_idx,2)' - filtered_diff(sorted_idx);

errorbar(1:length(sorted_idx), filtered_diff(sorted_idx), lower_error, upper_error, 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

ylabel('Scz - Ctl');
xticks(1:length(sorted_idx));
xticklabels(filtered_params(sorted_idx));
xtickangle(90);
title('DIP');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.FontSize = 14;

%% 
