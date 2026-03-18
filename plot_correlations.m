%% Figures - parameter correlations
cd 'Figures_DIP_DCM_25-main'
addpath('spm12/spm12') % Download SPm12

%% DCM 
% Load exp params
cd 'Figures_DIP_DCM_25-main/DCM'
load('DCM_LFP_PL_params.mat'); % 'params_pre_DCM_PL', 'cov_pre_DCM_PL', 'params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_params.mat'); % 'params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_params.mat'); % 'params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'params_post_DCM_LEV', 'cov_post_DCM_LEV');
% load log params
cd 'Figures_DIP_DCM_25-main/DCM'
load('DCM_LFP_PL_log_params_final.mat'); % 'log_params_pre_DCM_PL', 'cov_pre_DCM_PL', 'log_params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_log_params_final.mat'); % 'log_params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'log_params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_log_params_final.mat'); % 'log_params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'log_params_post_DCM_LEV', 'cov_post_DCM_LEV');

random_samples_pre_PL_DCM=zeros(500,15);
random_samples_post_PL_DCM=zeros(500,15);
random_samples_pre_LTG_DCM=zeros(500,15);
random_samples_post_LTG_DCM=zeros(500,15);
random_samples_pre_LEV_DCM=zeros(500,15);
random_samples_post_LEV_DCM=zeros(500,15);

for ii = 1:15
    mu_pre = log_params_pre_DCM_PL(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_PL(ii, 1));
    mu_post = log_params_post_DCM_PL(ii, 1);
    sigma_post = sqrt(cov_post_DCM_PL(ii, 1));
    N = 500;
    random_samples_pre_PL_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_PL_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end

for ii = 1:15
    mu_pre = log_params_pre_DCM_LTG(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LTG(ii, 1));
    mu_post = log_params_post_DCM_LTG(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LTG(ii, 1));
    N = 500;
    random_samples_pre_LTG_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LTG_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end
% cohen's d gives also the same reesult sampling and using the analytical
% formula
for ii = 1:15
    mu_pre = log_params_pre_DCM_LEV(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LEV(ii, 1));
    mu_post = log_params_post_DCM_LEV(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LEV(ii, 1));
    N = 500;
    random_samples_pre_LEV_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LEV_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end


%% DIP-DCM 
cd 'Figures_DIP_DCM_25-main/Hybrid'
load('params_PL_hybrid.mat');
load('log_params_PL_hybrid_posteriors.mat');
load('params_LTG_hybrid.mat');
load('log_params_LTG_hybrid_posteriors.mat');
load('params_LEV_hybrid.mat');
load('log_params_LEV_hybrid_posteriors.mat');
random_samples_pre_PL_DIP = zeros(500, 15); % Initialize the samples array
random_samples_post_PL_DIP = zeros(500, 15);

random_samples_pre_LTG_DIP = zeros(500, 15); % Initialize the samples array
random_samples_post_LTG_DIP = zeros(500, 15);

random_samples_pre_LEV_DIP = zeros(500, 15); % Initialize the samples array
random_samples_post_LEV_DIP = zeros(500, 15);

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


for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_PL_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_PL_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

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

for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LTG_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LTG_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end


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


for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LEV_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LEV_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

end

%%
R_dcm_pre_PL = corr(random_samples_pre_PL_DCM);
R_dcm_post_PL = corr(random_samples_post_PL_DCM);
R_dcm_pre_LTG = corr(random_samples_pre_LTG_DCM);
R_dcm_post_LTG = corr(random_samples_post_LTG_DCM);
R_dcm_pre_LEV = corr(random_samples_pre_LEV_DCM);
R_dcm_post_LEV = corr(random_samples_post_LEV_DCM);

R_dip_pre_PL = corr(random_samples_pre_PL_DIP);
R_dip_post_PL = corr(random_samples_post_PL_DIP);
R_dip_pre_LTG = corr(random_samples_pre_LTG_DIP);
R_dip_post_LTG = corr(random_samples_post_LTG_DIP);
R_dip_pre_LEV = corr(random_samples_pre_LEV_DIP);
R_dip_post_LEV = corr(random_samples_post_LEV_DIP);

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};

% figure;
% for i=1:15
%     for j=1:15
%     subplot(15,15,(i-1)*15 + j)        
%     scatter(random_samples_post_LEV_DIP(:,i), random_samples_post_LEV_DIP(:,j));
%     xlabel(paramsvec{i})
%     ylabel(paramsvec{j})
%     end
% end
% 
% figure;
% for i=1:15
%     for j=1:15
%     subplot(15,15,(i-1)*15 + j)        
%     scatter(random_samples_post_LEV_DCM(:,i), random_samples_post_LEV_DCM(:,j));
%     xlabel(paramsvec{i})
%     ylabel(paramsvec{j})
%     end
% end

originalCmap = [
    0.4, 0.7, 1.0;  % Sky Blue
    0.9, 0.7, 0.5;  % Light Brown/Peach
    0.9, 0.9, 0.9;  % Light Grey   
];
numColors = 50;
x = linspace(1, numColors, size(originalCmap, 1));
xi = linspace(1, numColors, numColors);
colors = [interp1(x, originalCmap(:,1), xi, 'linear')', ...
              interp1(x, originalCmap(:,2), xi, 'linear')', ...
              interp1(x, originalCmap(:,3), xi, 'linear')'];
colors= flipud(colors);
%%
R_dcm_pre_PL(1:size(R_dcm_pre_PL,1)+1:end)=0;
R_dcm_post_PL(1:size(R_dcm_post_PL,1)+1:end)=0;
R_dcm_pre_LTG(1:size(R_dcm_pre_LTG,1)+1:end) = 0;
R_dcm_post_LTG(1:size(R_dcm_post_LTG,1)+1:end)= 0;
R_dcm_pre_LEV(1:size(R_dcm_pre_LEV,1)+1:end) = 0;
R_dcm_post_LEV(1:size(R_dcm_post_LEV,1)+1:end)= 0;
R_dip_pre_PL(1:size(R_dip_pre_PL,1)+1:end)   = 0;
R_dip_post_PL(1:size(R_dip_post_PL,1)+1:end) = 0;
R_dip_pre_LTG(1:size(R_dip_pre_LTG,1)+1:end) = 0;
R_dip_post_LTG(1:size(R_dip_post_LTG,1)+1:end)= 0;
R_dip_pre_LEV(1:size(R_dip_pre_LEV,1)+1:end) = 0;
R_dip_post_LEV(1:size(R_dip_post_LEV,1)+1:end)= 0;

%%
CLIM = [-0.2 0.2];
figure;
ax1 = subplot(6,2,1);
imagesc(ax1,R_dcm_pre_PL); colormap(ax1,colors); colorbar; 
caxis(ax1, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax2 =subplot(6,2,2);
imagesc(ax2,R_dip_pre_PL);  colormap(ax2,colors); colorbar;
caxis(ax2, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;

ax3=subplot(6,2,3);
imagesc(ax3,R_dcm_post_PL);  colormap(ax3,colors); colorbar; 
caxis(ax3, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax4=subplot(6,2,4);
imagesc(ax4,R_dip_post_PL);  colormap(ax4,colors); colorbar; 
caxis(ax4, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;

ax5=subplot(6,2,5);
imagesc(ax5,R_dcm_pre_LTG);  colormap(ax5,colors); colorbar; 
caxis(ax5, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax6=subplot(6,2,6);
imagesc(ax6,R_dip_pre_LTG);  colormap(ax6,colors); colorbar; 
caxis(ax6, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax7=subplot(6,2,7);
imagesc(ax7,R_dcm_post_LTG);  colormap(ax7,colors); colorbar; 
caxis(ax7, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax8=subplot(6,2,8);
imagesc(ax8,R_dip_post_LTG);  colormap(ax8,colors); colorbar; 
caxis(ax8, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;

ax9=subplot(6,2,9);
imagesc(ax9,R_dcm_pre_LEV);  colormap(ax9,colors); colorbar; 
caxis(ax9, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax10=subplot(6,2,10);
imagesc(ax10,R_dip_pre_LEV);  colormap(ax10,colors); colorbar; 
caxis(ax10, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;

ax11=subplot(6,2,11);
imagesc(ax11,R_dcm_post_LEV);  colormap(ax11,colors); colorbar; 
caxis(ax11, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;
ax12=subplot(6,2,12);
imagesc(ax12,R_dip_post_LEV);  colormap(ax12,colors); colorbar; 
caxis(ax12, CLIM); 
hold on;
xticks(1:15);yticks(1:15);ytickangle(0)
xticklabels(paramsvec(1:15));
yticklabels(paramsvec(1:15));
ax = gca;ax.FontSize = 8;
hold on;

%% ============================================== %%
%% Correlations for the full model

%% DCM 
cd 'Figures_DIP_DCM_25-main/run_full_model_29_params'
load('DCM_LFP_PL_log_params_final.mat'); % 'log_params_pre_DCM_PL', 'cov_pre_DCM_PL', 'log_params_post_DCM_PL', 'cov_post_DCM_PL');
load('DCM_LFP_LTG_log_params_final.mat'); % 'log_params_pre_DCM_LTG', 'cov_pre_DCM_LTG', 'log_params_post_DCM_LTG', 'cov_post_DCM_LTG');
load('DCM_LFP_LEV_log_params_final.mat'); % 'log_params_pre_DCM_LEV', 'cov_pre_DCM_LEV', 'log_params_post_DCM_LEV', 'cov_post_DCM_LEV');

random_samples_pre_PL_DCM=zeros(500,17);
random_samples_post_PL_DCM=zeros(500,17);
random_samples_pre_LTG_DCM=zeros(500,17);
random_samples_post_LTG_DCM=zeros(500,17);
random_samples_pre_LEV_DCM=zeros(500,17);
random_samples_post_LEV_DCM=zeros(500,17);

for ii = 1:17
    mu_pre = log_params_pre_DCM_PL(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_PL(ii, 1));
    mu_post = log_params_post_DCM_PL(ii, 1);
    sigma_post = sqrt(cov_post_DCM_PL(ii, 1));
    N = 500;
    random_samples_pre_PL_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_PL_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end

for ii = 1:17
    mu_pre = log_params_pre_DCM_LTG(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LTG(ii, 1));
    mu_post = log_params_post_DCM_LTG(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LTG(ii, 1));
    N = 500;
    random_samples_pre_LTG_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LTG_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end
% cohen's d gives also the same reesult sampling and using the analytical
% formula
for ii = 1:17
    mu_pre = log_params_pre_DCM_LEV(ii, 1);
    sigma_pre = sqrt(cov_pre_DCM_LEV(ii, 1));
    mu_post = log_params_post_DCM_LEV(ii, 1);
    sigma_post = sqrt(cov_post_DCM_LEV(ii, 1));
    N = 500;
    random_samples_pre_LEV_DCM(:,ii) = mu_pre + sigma_pre * randn(N, 1);
    random_samples_post_LEV_DCM(:,ii) = mu_post + sigma_post * randn(N, 1);
end

%% DIP-DCM 
cd 'Figures_DIP_DCM_25-main/run_full_model_29_params'
load('params_PL_hybrid.mat');
load('log_params_PL_hybrid_posteriors.mat');
load('params_LTG_hybrid.mat');
load('log_params_LTG_hybrid_posteriors.mat');
load('params_LEV_hybrid.mat');
load('log_params_LEV_hybrid_posteriors.mat');
random_samples_pre_PL_DIP = zeros(500, 17); % Initialize the samples array
random_samples_post_PL_DIP = zeros(500, 17);

random_samples_pre_LTG_DIP = zeros(500, 17); % Initialize the samples array
random_samples_post_LTG_DIP = zeros(500, 17);

random_samples_pre_LEV_DIP = zeros(500, 17); % Initialize the samples array
random_samples_post_LEV_DIP = zeros(500, 17);

for ii=1:17
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


for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_PL_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_PL_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

end

for ii=1:17
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

for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LTG_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LTG_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end


end

for ii=1:17
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


for n = 1:500
    rand_num = rand(); % Generate uniform random numbser
    random_samples_pre_LEV_DIP(n,ii) = x(find(cumulative_density_pre >= rand_num, 1, 'first')); % Inverse transform sampling
    random_samples_post_LEV_DIP(n,ii) = x(find(cumulative_density_post >= rand_num, 1, 'first'));
end

end

%%
R_dcm_pre_PL = corr(random_samples_pre_PL_DCM);
R_dcm_post_PL = corr(random_samples_post_PL_DCM);
R_dcm_pre_LTG = corr(random_samples_pre_LTG_DCM);
R_dcm_post_LTG = corr(random_samples_post_LTG_DCM);
R_dcm_pre_LEV = corr(random_samples_pre_LEV_DCM);
R_dcm_post_LEV = corr(random_samples_post_LEV_DCM);

R_dip_pre_PL = corr(random_samples_pre_PL_DIP);
R_dip_post_PL = corr(random_samples_post_PL_DIP);
R_dip_pre_LTG = corr(random_samples_pre_LTG_DIP);
R_dip_post_LTG = corr(random_samples_post_LTG_DIP);
R_dip_pre_LEV = corr(random_samples_pre_LEV_DIP);
R_dip_post_LEV = corr(random_samples_post_LEV_DIP);

paramsvec = {'R1','R2','Te','Ti', 'Tk', 'He', 'Hi', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};

% figure;
% for i=1:15
%     for j=1:15
%     subplot(15,15,(i-1)*15 + j)        
%     scatter(random_samples_post_LEV_DIP(:,i), random_samples_post_LEV_DIP(:,j));
%     xlabel(paramsvec{i})
%     ylabel(paramsvec{j})
%     end
% end
% 
% figure;
% for i=1:15
%     for j=1:15
%     subplot(15,15,(i-1)*15 + j)        
%     scatter(random_samples_post_LEV_DCM(:,i), random_samples_post_LEV_DCM(:,j));
%     xlabel(paramsvec{i})
%     ylabel(paramsvec{j})
%     end
% end

originalCmap = [
    0.4, 0.7, 1.0;  % Sky Blue
    0.9, 0.7, 0.5;  % Light Brown/Peach
    0.9, 0.9, 0.9;  % Light Grey   
];
numColors = 50;
x = linspace(1, numColors, size(originalCmap, 1));
xi = linspace(1, numColors, numColors);
colors = [interp1(x, originalCmap(:,1), xi, 'linear')', ...
              interp1(x, originalCmap(:,2), xi, 'linear')', ...
              interp1(x, originalCmap(:,3), xi, 'linear')'];
colors= flipud(colors);
%%
R_dcm_pre_PL(1:size(R_dcm_pre_PL,1)+1:end)=0;
R_dcm_post_PL(1:size(R_dcm_post_PL,1)+1:end)=0;
R_dcm_pre_LTG(1:size(R_dcm_pre_LTG,1)+1:end) = 0;
R_dcm_post_LTG(1:size(R_dcm_post_LTG,1)+1:end)= 0;
R_dcm_pre_LEV(1:size(R_dcm_pre_LEV,1)+1:end) = 0;
R_dcm_post_LEV(1:size(R_dcm_post_LEV,1)+1:end)= 0;
R_dip_pre_PL(1:size(R_dip_pre_PL,1)+1:end)   = 0;
R_dip_post_PL(1:size(R_dip_post_PL,1)+1:end) = 0;
R_dip_pre_LTG(1:size(R_dip_pre_LTG,1)+1:end) = 0;
R_dip_post_LTG(1:size(R_dip_post_LTG,1)+1:end)= 0;
R_dip_pre_LEV(1:size(R_dip_pre_LEV,1)+1:end) = 0;
R_dip_post_LEV(1:size(R_dip_post_LEV,1)+1:end)= 0;

%%
CLIM = [-0.2 0.2];
figure;
ax1 = subplot(6,2,1);
imagesc(ax1,R_dcm_pre_PL); colormap(ax1,colors); colorbar; 
caxis(ax1, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax2 =subplot(6,2,2);
imagesc(ax2,R_dip_pre_PL);  colormap(ax2,colors); colorbar;
caxis(ax2, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;

ax3=subplot(6,2,3);
imagesc(ax3,R_dcm_post_PL);  colormap(ax3,colors); colorbar; 
caxis(ax3, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax4=subplot(6,2,4);
imagesc(ax4,R_dip_post_PL);  colormap(ax4,colors); colorbar; 
caxis(ax4, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;

ax5=subplot(6,2,5);
imagesc(ax5,R_dcm_pre_LTG);  colormap(ax5,colors); colorbar; 
caxis(ax5, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax6=subplot(6,2,6);
imagesc(ax6,R_dip_pre_LTG);  colormap(ax6,colors); colorbar; 
caxis(ax6, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax7=subplot(6,2,7);
imagesc(ax7,R_dcm_post_LTG);  colormap(ax7,colors); colorbar; 
caxis(ax7, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax8=subplot(6,2,8);
imagesc(ax8,R_dip_post_LTG);  colormap(ax8,colors); colorbar; 
caxis(ax8, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;

ax9=subplot(6,2,9);
imagesc(ax9,R_dcm_pre_LEV);  colormap(ax9,colors); colorbar; 
caxis(ax9, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax10=subplot(6,2,10);
imagesc(ax10,R_dip_pre_LEV);  colormap(ax10,colors); colorbar; 
caxis(ax10, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;

ax11=subplot(6,2,11);
imagesc(ax11,R_dcm_post_LEV);  colormap(ax11,colors); colorbar; 
caxis(ax11, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;
ax12=subplot(6,2,12);
imagesc(ax12,R_dip_post_LEV);  colormap(ax12,colors); colorbar; 
caxis(ax12, CLIM); 
hold on;
xticks(1:17);yticks(1:17);ytickangle(0)
xticklabels(paramsvec(1:17));
yticklabels(paramsvec(1:17));
ax = gca;ax.FontSize = 8;
hold on;

