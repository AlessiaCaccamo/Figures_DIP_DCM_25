%% Re-run analyses for 150 generations rather than 500.

%% Load model MOGA spectra
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LEV'

load('MOGA_LFP_model_spectra_LEV_grand_150_gen.mat'); % 'psd_m_pre_all', 'psd_m_post_all');
psd_m_pre_all_LEV=psd_m_pre_all;
psd_m_post_all_LEV=psd_m_post_all;
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/PL'
load('MOGA_LFP_model_spectra_PL_grand_150_gen.mat'); % 'psd_m_pre_all', 'psd_m_post_all');
psd_m_pre_all_PL=psd_m_pre_all;
psd_m_post_all_PL=psd_m_post_all;
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LTG'
load('MOGA_LFP_model_spectra_LTG_grand_150_gen.mat'); % 'psd_m_pre_all', 'psd_m_post_all');
psd_m_pre_all_LTG=psd_m_pre_all;
psd_m_post_all_LTG=psd_m_post_all;


%% Load MOGA params
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/PL'
load('MOGA_LFP_PL_params_150_gen.mat'); % 'all_sim_params_pre_PL', 'all_sim_params_post_PL');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LTG'
load('MOGA_LFP_LTG_params_150_gen.mat'); % 'all_sim_params_pre_LTG', 'all_sim_params_post_LTG');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LEV'
load('MOGA_LFP_LEV_params_150_gen.mat'); % 'all_sim_params_pre_LEV', 'all_sim_params_post_LEV');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/PL'
load('MOGA_LFP_PL_log_params_150_gen.mat'); % 'all_sim_log_params_pre_PL', 'all_sim_log_params_post_PL');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LTG'
load('MOGA_LFP_LTG_log_params_150_gen.mat'); % 'all_sim_log_params_pre_LTG', 'all_sim_log_params_post_LTG');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LEV'
load('MOGA_LFP_LEV_log_params_150_gen.mat'); % 'all_sim_log_params_pre_LEV', 'all_sim_log_params_post_LEV');

%% Load params obtained via hybrid method
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/PL/After_DCM'
load('params_PL_hybrid.mat');
load('log_params_PL_hybrid_posteriors.mat');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LTG/After_DCM'
load('params_LTG_hybrid.mat');
load('log_params_LTG_hybrid_posteriors.mat');

cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LEV/After_DCM'
load('log_params_LEV_hybrid_posteriors.mat');
load('params_LEV_hybrid.mat');

%% Hybrid models
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/PL/After_DCM'
load('models_PL_500_hybrid.mat', 'model_pre_PL_all', 'model_post_PL_all');
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LTG/After_DCM'
load('models_LTG_500_hybrid.mat', 'model_pre_LTG_all', 'model_post_LTG_all');
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all/LEV/After_DCM'
load('models_LEV_500_hybrid.mat', 'model_pre_LEV_all', 'model_post_LEV_all');


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


%% Calculate hybrid inferences as per script run_plot_spectra_params.m
for ii=1:15
muvec_pre=log_posteriors_pre_DCM_PL(ii,:);
varvec_pre=cov_pre_DCM_PL(ii, :);
muvec_post=log_posteriors_post_DCM_PL(ii,:);
varvec_post=cov_post_DCM_PL(ii, :);
x = linspace(min(muvec_pre) - 3*sqrt(max(varvec_pre)), max(muvec_pre) + 3*sqrt(max(varvec_pre)), 1000); % for log params
%subplot(3,5,ii);
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

random_samples_pre = zeros(1e5, 1); % Initialize the samples array
random_samples_post = zeros(1e5, 1);
for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre(n) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post(n) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end
%histogram(random_samples_pre, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5); % Histogram of samples
%histogram(random_samples_post, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5); % Histogram of samples
d_PL(ii)=(mean(random_samples_post) - mean(random_samples_pre))/sqrt((var(random_samples_post)+var(random_samples_pre))/2);
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

random_samples_pre = zeros(1e5, 1); % Initialize the samples array
random_samples_post = zeros(1e5, 1);
for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre(n) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post(n) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end
%histogram(random_samples_pre, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5); % Histogram of samples
%histogram(random_samples_post, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5); % Histogram of samples
d_LTG(ii)=(mean(random_samples_post) - mean(random_samples_pre))/sqrt((var(random_samples_post)+var(random_samples_pre))/2);
end

for ii=1:15
muvec_pre=log_posteriors_pre_DCM_LEV(ii,:);
varvec_pre=cov_pre_DCM_LEV(ii, :);
muvec_post=log_posteriors_post_DCM_LEV(ii,:);
varvec_post=cov_post_DCM_LEV(ii, :);
x = linspace(min(muvec_pre) - 3*sqrt(max(varvec_pre)), max(muvec_pre) + 3*sqrt(max(varvec_pre)), 1000); % for log params
%subplot(3,5,ii);
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

random_samples_pre = zeros(1e5, 1); % Initialize the samples array
random_samples_post = zeros(1e5, 1);
for n = 1:1e5
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre(n) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post(n) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end
%histogram(random_samples_pre, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5); % Histogram of samples
%histogram(random_samples_post, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5); % Histogram of samples
d_LEV(ii)=(mean(random_samples_post) - mean(random_samples_pre))/sqrt((var(random_samples_post)+var(random_samples_pre))/2);
end

%% Inferences summary Cohen's d 
% cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Placebo/LFP_DCM/Hybrid_PL_efficiency/150_gen_all'
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_reduced_150_generations'
load('inferences_summary_data_150_gen.mat'); % load inferences from file (they are the same as those generated above)
d_PL_2=d_PL';
d_LTG_2=d_LTG';
d_LEV_2=d_LEV';


%% Raw effects

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




%% re-save different numbers of repeats 
addpath '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_reduced_150_generations'

load('log_PL_spectra.mat');
load('log_LTG_spectra.mat');
load('log_LEV_spectra.mat');
for sim = 1:1000
    RMSE_MOGA_LEV_pre(1,sim) = sqrt(mean((data_psd_pre_LEV(1:57) - psd_m_pre_all_LEV(1:57,sim)).^2));
    RMSE_MOGA_LEV_post(1,sim) = sqrt(mean((data_psd_post_LEV(1:57) - psd_m_post_all_LEV(1:57,sim)).^2));
    RMSE_MOGA_LTG_pre(1,sim) = sqrt(mean((data_psd_pre_LTG(1:57) - psd_m_pre_all_LTG(1:57,sim)).^2));
    RMSE_MOGA_LTG_post(1,sim) = sqrt(mean((data_psd_post_LTG(1:57) - psd_m_post_all_LTG(1:57,sim)).^2));
    RMSE_MOGA_PL_pre(1,sim) = sqrt(mean((data_psd_pre_PL(1:57) - psd_m_pre_all_PL(1:57,sim)).^2));
    RMSE_MOGA_PL_post(1,sim) = sqrt(mean((data_psd_post_PL(1:57) - psd_m_post_all_PL(1:57,sim)).^2));
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

i = [50 100 150 200 250 300 350 400 450 500]; 
n_values = numel(i); 

diff_sim_numbers = cell(n_values, 6);
pre_PL_MOGA_params = cell(n_values, 1);
post_PL_MOGA_params = cell(n_values, 1);
pre_LTG_MOGA_params = cell(n_values, 1);
post_LTG_MOGA_params = cell(n_values, 1);
pre_LEV_MOGA_params = cell(n_values, 1);
post_LEV_MOGA_params = cell(n_values, 1);

for idx = 1:n_values
    n = i(idx); 
    [RMSE_LEV_pre_2, indices_LEV_pre_2] = mink(RMSE_LEV_pre, n);
    [RMSE_LEV_post_2, indices_LEV_post_2] = mink(RMSE_LEV_post, n);
    [RMSE_LTG_pre_2, indices_LTG_pre_2] = mink(RMSE_LTG_pre, n);
    [RMSE_LTG_post_2, indices_LTG_post_2] = mink(RMSE_LTG_post, n);
    [RMSE_PL_pre_2, indices_PL_pre_2] = mink(RMSE_PL_pre, n);
    [RMSE_PL_post_2, indices_PL_post_2] = mink(RMSE_PL_post, n);

    diff_sim_numbers{idx, 1} = indices_LEV_pre_2;
    diff_sim_numbers{idx, 2} = indices_LEV_post_2;
    diff_sim_numbers{idx, 3} = indices_LTG_pre_2;
    diff_sim_numbers{idx, 4} = indices_LTG_post_2;
    diff_sim_numbers{idx, 5} = indices_PL_pre_2;
    diff_sim_numbers{idx, 6} = indices_PL_post_2;

    pre_PL_MOGA_params{idx} = all_sim_log_params_pre_PL(:, diff_sim_numbers{idx, 5}');
    post_PL_MOGA_params{idx} = all_sim_log_params_post_PL(:, diff_sim_numbers{idx, 6}');
    pre_LTG_MOGA_params{idx} = all_sim_log_params_pre_LTG(:, diff_sim_numbers{idx, 3}');
    post_LTG_MOGA_params{idx} = all_sim_log_params_post_LTG(:, diff_sim_numbers{idx, 4}');
    pre_LEV_MOGA_params{idx} = all_sim_log_params_pre_LEV(:, diff_sim_numbers{idx, 1}');
    post_LEV_MOGA_params{idx} = all_sim_log_params_post_LEV(:, diff_sim_numbers{idx, 2}');
end

%% Load the corresponding hybrid repeats

i = [50 100 150 200 250 300 350 400 450 500]; 
n_values = numel(i); 
for idx = 1:n_values
    n = i(idx);     
    eval(['params_pre_DCM_PL_' num2str(n) ' = params_pre_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 5});']);
    eval(['params_post_DCM_PL_' num2str(n) ' = params_post_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 6});']);
    eval(['params_pre_DCM_LTG_' num2str(n) ' = params_pre_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 3});']);
    eval(['params_post_DCM_LTG_' num2str(n) ' = params_post_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 4});']);
    eval(['params_pre_DCM_LEV_' num2str(n) ' = params_pre_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 1});']);
    eval(['params_post_DCM_LEV_' num2str(n) ' = params_post_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 2});']);
    eval(['log_posteriors_pre_DCM_PL_' num2str(n) ' = log_posteriors_pre_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 5});']);
    eval(['log_posteriors_post_DCM_PL_' num2str(n) ' = log_posteriors_post_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 6});']);
    eval(['log_posteriors_pre_DCM_LTG_' num2str(n) ' = log_posteriors_pre_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 3});']);
    eval(['log_posteriors_post_DCM_LTG_' num2str(n) ' = log_posteriors_post_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 4});']);
    eval(['log_posteriors_pre_DCM_LEV_' num2str(n) ' = log_posteriors_pre_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 1});']);
    eval(['log_posteriors_post_DCM_LEV_' num2str(n) ' = log_posteriors_post_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 2});']);
    eval(['cov_pre_DCM_PL_' num2str(n) ' = cov_pre_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 5});']);
    eval(['cov_post_DCM_PL_' num2str(n) ' = cov_post_DCM_PL(:, diff_sim_numbers{' num2str(idx) ', 6});']);
    eval(['cov_pre_DCM_LTG_' num2str(n) ' = cov_pre_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 3});']);
    eval(['cov_post_DCM_LTG_' num2str(n) ' = cov_post_DCM_LTG(:, diff_sim_numbers{' num2str(idx) ', 4});']);
    eval(['cov_pre_DCM_LEV_' num2str(n) ' = cov_pre_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 1});']);
    eval(['cov_post_DCM_LEV_' num2str(n) ' = cov_post_DCM_LEV(:, diff_sim_numbers{' num2str(idx) ', 2});']);
end


%% Replot inferences for those selected parameters & see when it breaks

prefixes = {'PL', 'LTG', 'LEV'};
n_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
num_samples = 1e5; % Number of random samples

for p = 1:length(prefixes)
    prefix = prefixes{p};
    for n = n_values
        for ii = 1:15
            % Construct variable names dynamically
            muvec_pre = eval(['log_posteriors_pre_DCM_' prefix '_' num2str(n) '(ii,:)']);
            varvec_pre = eval(['cov_pre_DCM_' prefix '_' num2str(n) '(ii,:)']);
            muvec_post = eval(['log_posteriors_post_DCM_' prefix '_' num2str(n) '(ii,:)']);
            varvec_post = eval(['cov_post_DCM_' prefix '_' num2str(n) '(ii,:)']);
  
            x = linspace(min(muvec_pre) - 3*sqrt(max(varvec_pre)), max(muvec_pre) + 3*sqrt(max(varvec_pre)), 1000);
            total_density_pre = zeros(size(x));
            total_density_post = zeros(size(x));
            
            for i = 1:length(muvec_pre)
                mu_pre = muvec_pre(i);
                sigma_pre = sqrt(varvec_pre(i));
                mu_post = muvec_post(i);
                sigma_post = sqrt(varvec_post(i));
                pdf_pre = (1/(sigma_pre * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_pre) / sigma_pre).^2);
                pdf_post = (1/(sigma_post * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_post) / sigma_post).^2);
                total_density_pre = total_density_pre + (pdf_pre/n);
                total_density_post = total_density_post + (pdf_post/n);
            end
            
            cumulative_density_pre = cumsum(total_density_pre) / sum(total_density_pre);
            cumulative_density_post = cumsum(total_density_post) / sum(total_density_post);
            
            random_samples_pre = zeros(num_samples, 1);
            random_samples_post = zeros(num_samples, 1);
            
            for n_rep = 1:num_samples
                rand_num = rand();
                random_samples_pre(n_rep) = x(find(cumulative_density_pre >= rand_num, 1, 'first'));
                random_samples_post(n_rep) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
            end
            
            eval(['d_' prefix '_' num2str(n) '(ii) = (mean(random_samples_post) - mean(random_samples_pre)) / sqrt((var(random_samples_post) + var(random_samples_pre)) / 2);']);
        end
    end
end


% save('inferences_summary_data_all_gen.mat', 'd_PL_50', 'd_PL_100','d_PL_150','d_PL_200','d_PL_250','d_PL_300','d_PL_350','d_PL_400','d_PL_450','d_PL_500', ...
%     'd_LTG_100','d_LTG_150','d_LTG_200','d_LTG_250','d_LTG_300','d_LTG_350','d_LTG_400','d_LTG_450','d_LTG_500',...
%     'd_LEV_100','d_LEV_150','d_LEV_200','d_LEV_250','d_LEV_300','d_LEV_350','d_LEV_400','d_LEV_450','d_LEV_500');



%%
prefixes = {'PL', 'LTG', 'LEV'};
n_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
num_samples = 1e5; % Number of random samples

for p = 1:length(prefixes)
    prefix = prefixes{p};
    for n = n_values
        for ii = 1:15
            % Construct variable names dynamically
            muvec_pre = eval(['log_posteriors_pre_DCM_' prefix '_' num2str(n) '(ii,:)']);
            varvec_pre = eval(['cov_pre_DCM_' prefix '_' num2str(n) '(ii,:)']);
            muvec_post = eval(['log_posteriors_post_DCM_' prefix '_' num2str(n) '(ii,:)']);
            varvec_post = eval(['cov_post_DCM_' prefix '_' num2str(n) '(ii,:)']);
  
            x = linspace(min(muvec_pre) - 3*sqrt(max(varvec_pre)), max(muvec_pre) + 3*sqrt(max(varvec_pre)), 1000);
            total_density_pre = zeros(size(x));
            total_density_post = zeros(size(x));
            
            for i = 1:length(muvec_pre)
                mu_pre = muvec_pre(i);
                sigma_pre = sqrt(varvec_pre(i));
                mu_post = muvec_post(i);
                sigma_post = sqrt(varvec_post(i));
                pdf_pre = (1/(sigma_pre * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_pre) / sigma_pre).^2);
                pdf_post = (1/(sigma_post * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_post) / sigma_post).^2);
                total_density_pre = total_density_pre + (pdf_pre/n);
                total_density_post = total_density_post + (pdf_post/n);
            end
            
            cumulative_density_pre = cumsum(total_density_pre) / sum(total_density_pre);
            cumulative_density_post = cumsum(total_density_post) / sum(total_density_post);
            
            random_samples_pre = zeros(num_samples, 1);
            random_samples_post = zeros(num_samples, 1);
            
            for n_rep = 1:num_samples
                rand_num = rand();
                random_samples_pre(n_rep) = x(find(cumulative_density_pre >= rand_num, 1, 'first'));
                random_samples_post(n_rep) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
            end
            
            eval(['D_DIP_' prefix '_' num2str(n) '(ii) = mean(random_samples_post - random_samples_pre);']);
            eval(['BCI_DIP_' prefix '_' num2str(n) '(ii,:) = prctile(random_samples_post - random_samples_pre, [2.5, 97.5]);']);
        end
    end
end


% save('inferences_summary_data_all_gen_final.mat', 'D_DIP_PL_50', 'D_DIP_PL_100','D_DIP_PL_150','D_DIP_PL_200','D_DIP_PL_250','D_DIP_PL_300','D_DIP_PL_350','D_DIP_PL_400','D_DIP_PL_450','D_DIP_PL_500', ...
%     'D_DIP_LTG_100','D_DIP_LTG_150','D_DIP_LTG_200','D_DIP_LTG_250','D_DIP_LTG_300','D_DIP_LTG_350','D_DIP_LTG_400','D_DIP_LTG_450','D_DIP_LTG_500',...
%     'D_DIP_LEV_100','D_DIP_LEV_150','D_DIP_LEV_200','D_DIP_LEV_250','D_DIP_LEV_300','D_DIP_LEV_350','D_DIP_LEV_400','D_DIP_LEV_450','D_DIP_LEV_500');
% 

%% Supplementary Figure S2B
cd '/Users/alessiacaccamo/Documents/Exeter/Data/TMS-EEG-Biondi2022/TMS-EEG_Isabella/TMS-EEG_Isabella/eyes_closed/Figures_DIP_DCM_25/run_reduced_150_generations'
load('inferences_summary_data_all_gen.mat');
load('inferences_summary_data_all_gen_final.mat');

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
n_values = [500, 450, 400, 350, 300, 250, 200, 150, 100, 50];  % Values for n

figure;
for idx = 1:length(n_values)
    n = n_values(idx);
    subplot(2, 5, idx);
    title(num2str(n_values));
    hold on;


    eval(['data_d_PL_hyb= d_PL_' num2str(n) ''';']);
    eval(['data_d_LTG_hyb= d_LTG_' num2str(n) ''';']);
    eval(['data_d_LEV_hyb= d_LEV_' num2str(n) ''';']);

    eval(['data_PL_hyb= D_DIP_PL_' num2str(n) ''';']);
    eval(['data_LTG_hyb= D_DIP_LTG_' num2str(n) ''';']);
    eval(['data_LEV_hyb= D_DIP_LEV_' num2str(n) ''';']);

    eval(['BCI_PL= BCI_DIP_PL_' num2str(n) ''';']);
    eval(['BCI_LTG= BCI_DIP_LTG_' num2str(n) ''';']);
    eval(['BCI_LEV= BCI_DIP_LEV_' num2str(n) ''';']);
    % 
    % data_D = [data_PL_hyb, data_LTG_hyb, data_LEV_hyb];
    % BCI_data = cat(3, BCI_PL', BCI_LTG', BCI_LEV');
    % data_d = [data_d_PL_hyb, data_d_LTG_hyb, data_d_LEV_hyb];
    % eval(['treatments_methods = {''PL,DIP-' num2str(n) ''', ''LTG,DIP-' num2str(n) ''', ''LEV,DIP-' num2str(n) '''};']);
    % 
    valid_idx = ...
    (abs(data_d_PL_hyb) >= 0.2) & ...                      
    ((BCI_PL(1,:)' > 0 & BCI_PL(2,:)' > 0) | ...       
    (BCI_PL(1,:)' < 0 & BCI_PL(2,:)' < 0) );          

    filtered_mu_diff = data_PL_hyb(valid_idx)';
    BCI_PL=BCI_PL';
    filtered_BCI = BCI_PL(valid_idx', :);
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
end



