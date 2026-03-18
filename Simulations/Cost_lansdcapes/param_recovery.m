%% Compare the estimated parameters to the real parameters used to generate the synthetic psd
%% ========================= Standard DCM =====================
% Example:
cd 'Figures_DIP_DCM_25-main/Simulations/Cost_lansdcapes/R1_Di'
load('standard_DCM_synthetic_spec_fixed.mat', 'DCM_all');
%load('wider3Var_DCM_synthetic_spec_fixed.mat', 'DCM_all')
%load('MidBoundsPriors_DCM_synthetic_spec_fixed.mat', 'DCM_all')

load('synthetic_psd_params_R1Di.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;
P=[params_1,params_2];

for dat=1:2
mu_true(:,dat)=P(:,dat);
end 

mu_estim=zeros(41,2);
cov_estim=zeros(41,2);
spec_estim_real=zeros(57,2);

for dat=1:2
DCM_all_selected=DCM_all(1,dat);    
mu_estim(:,dat)=spm_vec(DCM_all_selected{1}.Ep);
cov_estim(:,dat)=diag(DCM_all_selected{1}.Cp);
spec_estim=spm_csd_mtf(DCM_all_selected{1}.Ep,DCM_all_selected{1}.M, DCM_all_selected{1}.xU);
spec_estim_real(:,dat)=real(spec_estim{1});
end 

%%
figure;
for dat=1
plot(freq,spec(:,dat), '--', 'LineWidth',3,'Color','k');
hold on;
plot(freq,spec_estim_real(:,dat), 'LineWidth',2,'Color',[1.00 0.55 0.55]);
hold on;
xlabel('Frequency (Hz)');
ylabel('Power log_{10}(\muV^2/Hz)');
ax = gca; % Get current axis
ax.FontSize=24;
end 
legend('True', 'Estimated');

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
% hold on;

%% Check parameters recovered 

% Plot distributions in log space 
for dat=1
figure;
positions = [1 15]; % Parameters R1 and Di
for idx = 1:length(positions)
    i = positions(idx); % Retrieve the correct index (from positions array)
    subplot(2,2, idx); %
    hold on;
    muvec_pre = mu_true(i,:); % DIP prior
    muvec_post = mu_estim(i, dat); % DIP prior
    varvec_post = sqrt(cov_estim(i, dat)); % DIP variance of 1
    x_post = linspace(muvec_post - 4*varvec_post, muvec_post + 4*varvec_post, 200);
    y_post = normpdf(x_post, muvec_post, varvec_post); 
    plot(x_post, y_post, 'LineWidth', 2, 'Color', 'k');
    hold on;
    xline(muvec_pre(:,1),'--', 'LineWidth', 2, 'Color', [1.00 0.55 0.55]);
    hold on;
    xline(muvec_pre(:,2),'--', 'LineWidth', 2, 'Color', [0.35 0.85 0.75]);
    hold on;
    
    lower = muvec_post - 1.96*varvec_post;   
    upper = muvec_post + 1.96*varvec_post; 
    %xline(lower);
    %xline(upper);

    xlabel('Parameter Value');
    ylabel('Density');
    title(paramsvec{i});
    ax = gca; % Get current axis
    ax.XAxis.Exponent = 0; % Disable scientific notation
    ax.FontSize=16;
    % legend('Estimated', 'True 1', 'True 2', '95% CI')
    legend('Estimated', 'True 1', 'True 2')
    hold on;
end
end


%% ====================== DIP ===================================
%% Load results 
load('synthetic_psd_params_R1Di.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;
P=[params_1,params_2];

% This can only be run using the raw data:

% select_num=500;
% for i=1:2
%     for nsim=1:select_num 
%     load(['DIP_synthetic_spec_fixed_params/DCM_LFP_dataset_' num2str(i) '_nsim_' num2str(nsim) '.mat'], 'DCM');
%     DCM_all_dip(nsim,i)=DCM;
%     end 
% end

  
load('DCM_all_dip.mat');

%% Plot spectra

load('synthetic_psd_params_R1Di.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;
P=[params_1,params_2];

for dat=1:2
mu_true(:,dat)=P(:,dat);
end 

spec_estim_real_dip=zeros(57,500,2);
mu_estim_dip=zeros(44,500,2);
cov_estim_dip=zeros(44,500,2);
mu_estim_ga=zeros(44,500,2);

for nsim=1:500
for dat=1:2
DCM_all_selected_dip=DCM_all_dip(nsim,dat);  
temp = spm_vec(DCM_all_selected_dip.Ep);   
temp(14) = [];                              % remove 14th element, which is the additional parameter C that is zero, so that the 15th is Di.
mu_estim_dip(:,nsim,dat) = temp;            
temp_cov=diag(DCM_all_selected_dip.Cp);
temp_cov(14) = [];                              % remove 14th element, which is the additional parameter C that is zero, so that the 15th is Di.
cov_estim_dip(:,nsim,dat)=temp_cov;
spec_estim_dip=spm_csd_mtf(DCM_all_selected_dip.Ep,DCM_all_selected_dip.M, DCM_all_selected_dip.xU);
spec_estim_real_dip(:,nsim,dat)=real(spec_estim_dip{1});
temp_ga=spm_vec(DCM_all_selected_dip.M.pE);
temp_ga(14) = [];       
mu_estim_ga(:,nsim,dat)=temp_ga;
end 
end
%%
figure;
for dat=1
plot(freq,spec(:,dat), '--', 'LineWidth',2,'Color','k');
hold on;
plot(freq,mean(spec_estim_real_dip(:,:,dat),2), 'LineWidth',1,'Color',[1.00 0.55 0.55]);
hold on;
xlabel('Frequency (Hz)');
ylabel('Power log_{10}(\muV^2/Hz)');
ax = gca; % Get current axis
ax.FontSize=16;
end 
legend('True', 'Estimated');

%% Plot params
for dat=1
figure;
positions = [1 15]; 
for Idx = 1:length(positions)
ii = positions(Idx);
subplot(2,2, Idx);
muvec_pre = mu_true(ii,:); 
muvec_post=mu_estim_dip(ii,:,dat);
varvec_post=cov_estim_dip(ii,:,dat);
x_post = linspace(min(muvec_post) - 3*sqrt(max(varvec_post)), max(muvec_post) + 3*sqrt(max(varvec_post)), 1000); % for log params
hold on;
total_density_post = zeros(size(x_post));
for i = 1:length(muvec_post)
    hold on;
    mu_post = muvec_post(:,i);
    sigma_post = sqrt(varvec_post(:,i)); % Standard deviation is the square root of variance
    pdf_post = (1/(sigma_post * sqrt(2 * pi))) * exp(-0.5 * ((x_post - mu_post) / sigma_post).^2);% pdf of current distribution
    total_density_post = total_density_post + (pdf_post/500);
end
hold on;
xline(muvec_pre(:,1),'--', 'LineWidth', 2, 'Color', [1.00 0.55 0.55]);
hold on;
xline(muvec_pre(:,2),'--', 'LineWidth', 2, 'Color', [0.35 0.85 0.75]);
hold on;
plot(x_post, total_density_post, 'k', 'LineWidth', 2); 
xlabel('Parameter Value');
ylabel('Density');
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3', 'De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
title(paramsvec{ii})
ax = gca;               
ax.FontSize = 16;
legend('True 1', 'True 2', 'Estimated')
hold off;
end
end





