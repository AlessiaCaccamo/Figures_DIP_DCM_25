cd 'Figures_DIP_DCM_25-main/Simulations/Cost_lansdcapes'
%  Cost landscapes, LFP model 
%  Each figure shows error between model PSD and empirical PSD
%  across a 2D grid of pairwise parameters.

% PSD of data for pre-LEV
load('log_PL_spectra.mat');  % load PSD
data_spec=data_psd_pre_PL;
freq = (2:0.5:30)';

% Parameter bounds 
lb = [-1.4142 -1.4142 -1.4142 -1.4142 -1.1068 -1.0046 -1.3208 -1.0046 -1.0046 -1.0846 -3.2533 -2.8284 -3.2769 -1.0000 -0.7071 -0.3536 -0.5851 -0.3774 -0.3536 -0.3774 -0.3536 -0.3634 -0.3536 -0.3536 -0.4002 -0.5000 -0.6625]'; 
ub = [1.4142 1.4142 1.4142 1.4142 1 1 1 1.0509 1.0006 1.1017 2.8284 2.9713 2.8284 1 0.7768 0.3536 0.3536 0.3536 0.3562 0.3536 0.3562 0.3536 0.3712 0.4410 0.3536 0.5000 0.5000]';

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'}';

load('params.mat', 'params');
%%
load('synthetic_psd_params_R1Di.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;
P=[params_1,params_2];

cmap = colormap('default');          
amount = 0.2;            
cmap = cmap*(1-amount) + amount*ones(size(cmap));  

N = 200; % number of grid points per param
% pairwise parameter grid
for i = 1
    load("params.mat");

    figure;
    cost_all = zeros(N,N,15);
    p1_range_all=zeros(N,15);
    p2_range_all=zeros(N,15);
    for j = i+14
        p1 = paramsvec{i};
        p2 = paramsvec{j};

        % Create grid
        p1_range = linspace(lb(i), ub(i), N);
        p2_range = linspace(lb(j), ub(j), N);

        cost = zeros(N,N);
        for ii = 1:N
            for jj = 1:N
                % Copy base parameters
                p = params;
                % Replace varying parameters
                p(i) = p1_range(ii);
                p(j) = p2_range(jj);      
                model_spec=generate_spectrum(p,freq); % PSD of model
                cost(ii,jj) = sqrt(mean((model_spec - model_spec_1).^2)); % Compute cost 
            end
        end
        cost_all(:,:,j)=cost;
        p1_range_all(:,j)=p1_range;
        p2_range_all(:,j)=p2_range;
        % Plot landscape
        subplot(3, 5, j-1);
        imagesc(p1_range, p2_range, cost');
        set(gca, 'YDir', 'normal', 'FontSize',16);
        colormap(cmap);
        colorbar;
        xlabel(p1, 'Interpreter','none', 'FontSize', 16);
        ylabel(p2, 'Interpreter','none', 'FontSize', 16);
    end
   save(['cost_landscape_200_fixedP_R1Di_' num2str(i) '.mat'],"p1_range_all", "p2_range_all", 'cost_all');
end

%% Plot spectra corresponding to good fitting regions
load('cost_landscape_200_fixedP_R1Di_1.mat')
A = cost_all(:,:,15);       % extract the 4th layer% Sort all values and keep original indices
[sorted_vals, sorted_idx] = sort(A(:));% Two smallest values
[rows, cols] = ind2sub(size(A), sorted_idx(1:2));

%%
originalCmap = [
    0.2, 0.2, 0.6   % Dark Blue
    0.4, 0.7, 1.0;  % Sky Blue
    0.9, 0.7, 0.5;  % Light Brown/Peach
    0.9, 0.9, 0.9;  % Light Grey   
];
numColors = 500;
x = linspace(1, numColors, size(originalCmap, 1));
xi = linspace(1, numColors, numColors);
cmap = [interp1(x, originalCmap(:,1), xi, 'linear')', ...
              interp1(x, originalCmap(:,2), xi, 'linear')', ...
              interp1(x, originalCmap(:,3), xi, 'linear')'];


for i = 1
    figure;
    load(['cost_landscape_200_fixedP_R1Di_' num2str(i)]);
    for j = i+14
        p1 = paramsvec{i};
        p2 = paramsvec{j};
        imagesc(p1_range_all(:,j), p2_range_all(:,j), squeeze(cost_all(:,:,j))');
        hold on;
        scatter(p1_range_all(93,j), p2_range_all(142,j),100, 'filled','o','MarkerFaceColor',[1.00 0.55 0.55], 'MarkerEdgeColor','k');
        hold on;
        scatter(p1_range_all(151,j), p2_range_all(142,j),100, 'filled','o','MarkerFaceColor',[0.35 0.85 0.75], 'MarkerEdgeColor','k');
        set(gca, 'YDir', 'normal', 'FontSize',16);
        colormap(cmap);
        colorbar;
        xlabel(p1, 'Interpreter','none', 'FontSize', 24);
        ylabel(p2, 'Interpreter','none', 'FontSize', 24);

    end
end
        
%%
load('params');
p = params;
for i = 1 
    figure;
    load(['cost_landscape_200_fixedP_R1Di_' num2str(i)]);
    subplot(2,2,1);
    for j = i+14
        p(i) = p1_range_all(93,j);
        p(j) = p2_range_all(142,j); 
        params_1=p;
        model_spec_1=generate_spectrum(p,freq);
        plot(freq, model_spec_1, 'linewidth',2, 'color', [1.00 0.55 0.55]);
        hold on;
        set(gca, 'YDir', 'normal', 'FontSize',16);
        xlabel('Frequency (Hz)', 'FontSize', 16);
        ylabel('Power log_{10}(\muV^2/Hz)', 'FontSize', 16);
        %title('Synthetic data ');
    end
end
hold on;
for i = 1
    load(['cost_landscape_200_fixedP_R1Di_' num2str(i)]);
    subplot(2,2,2);
    for j = i+14
        p(i) = p1_range_all(151,j);
        p(j) = p2_range_all(142,j);   
        params_2=p;
        model_spec_2=generate_spectrum(p,freq);
        %plot(freq, data_spec,'--', 'linewidth',2, 'color', 'k');
        hold on;
        plot(freq, model_spec_2, '-', 'linewidth',2, 'color', [0.35 0.85 0.75]);
        set(gca, 'YDir', 'normal', 'FontSize',16);
        xlabel('Frequency (Hz)', 'FontSize', 16);
        ylabel('Power log_{10}(\muV^2/Hz)', 'FontSize', 16);
        %title('Synthetic data ');
        box on;
    end
end

%%
cd 'Cost_landscapes/R1_De'

for i = 1
    figure;
    load(['cost_landscape_200_fixedP_R1De_' num2str(i)]);
    for j = i+13
        p1 = paramsvec{i};
        p2 = paramsvec{j};
        imagesc(p1_range_all(:,j), p2_range_all(:,j), squeeze(cost_all(:,:,j))');
        hold on;
        scatter(p1_range_all(92,j), p2_range_all(5,j),100, 'filled','o','MarkerFaceColor',[1.00 0.55 0.55], 'MarkerEdgeColor','k');
        hold on;
        scatter(p1_range_all(151,j), p2_range_all(5,j),100, 'filled','o','MarkerFaceColor',[0.35 0.85 0.75], 'MarkerEdgeColor','k');
        set(gca, 'YDir', 'normal', 'FontSize',16);
        colormap(cmap);
        colorbar;
        xlabel(p1, 'Interpreter','none', 'FontSize', 20);
        ylabel(p2, 'Interpreter','none', 'FontSize', 20);

    end
end
