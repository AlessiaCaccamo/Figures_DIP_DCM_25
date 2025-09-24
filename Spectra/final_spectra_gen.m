%% Generate and plot spectra
%cd 'C:\Users\aleso\OneDrive\Desktop\Exeter\Data\TMS-EEG-Biondi2022\TMS-EEG_Isabella\TMS-EEG_Isabella\eyes_closed\Spectra'
%%
% PL
load('patient_data_Placebo.mat'); % data is not provided for ethical reasons
figure; 

%gaussian_kernel = fspecial('gaussian', [1 6], 5);  % 1 5 2

% sigma = 6;  % std Gaussian window
% window_size = 10;  % size Gaussian window
% gaussian_window = (exp(-((1:window_size) - (window_size+1)/2).^2 / (2*sigma^2)))/(sigma*sqrt(2*pi));
% gaussian_window = gaussian_window / sum(gaussian_window);  % Normalise

for ii = 1:14    
    ii
    subplot(3,5,ii)
    D = Data_Placebo(ii).pre.DataOut;
    average_spectrum = [];
    average_spectrum_post = [];
    for channel_n = 1:64 
        channel_n
    for i = 1:length(D.trial) % Iterate through trials and channels, for each individual subject  
            x = D.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            baseline = mean(D.trial{i},1);
            x = x - baseline;
            N = length(x);
            f_sampling = 1000; 
            f = (0:N-1) * f_sampling / N;
            X = fft(x);
            Pxx = (1/(N*f_sampling)) * abs(X).^2;
            indx=find(f>=2 & f<=45);
            Pxx =Pxx(indx);
            f = f(indx);

            %Pxx = conv(Pxx, gaussian_window, 'same'); % Smooth power spectrum
            
            Pxx_2 = Pxx/trapz(f,Pxx);
            %Pxx_2=imfilter(Pxx,gaussian_kernel);     
            average_spectrum = [average_spectrum; Pxx_2];      
    end
    mas = mean(average_spectrum, 1); % mean for trial and channel    
    all_channels_pre(channel_n,:)=mas;
    % Post-drug
    D_post = Data_Placebo(ii).post.DataOut;
    for i = 1:length(D_post.trial) % Iterate through trials and channels, for each individual subject  
            x_post = D_post.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            % Subtract mean for each trial
            baseline_post = mean(D_post.trial{i},1);
            x_post = x_post - baseline_post;
            N_post = length(x_post);
            f_sampling = 1000; 
            f_post = (0:N-1) * f_sampling / N_post;
            X_post = fft(x_post);
            Pxx_post = (1/(N_post*f_sampling)) * abs(X_post).^2;

            indx_post=find(f_post>=2 & f_post<=45);
            Pxx_post =Pxx_post(indx_post);
            f_post = f_post(indx_post);
            
            %Pxx_post = conv(Pxx_post, gaussian_window, 'same'); % Smooth power spectrum

            Pxx_post_2 = Pxx_post/trapz(f_post,Pxx_post);
            average_spectrum_post = [average_spectrum_post; Pxx_post_2];
    end
    mas_post = mean(average_spectrum_post, 1); % mean for trial and channel
    all_channels_post(channel_n,:)=mas_post;
    end
    all_subject_spectra_pre_PL(ii, :,:) = mas; % Spectrum of 1 subject pre treatment
    all_subject_spectra_post_PL(ii, :,:) = mas_post; % Spectrum of 1 subject post treatment
    mas_pre_std = std(mas, 0, 2);
    mas_post_std = std(mas_post, 0, 2); 
    plot(f,mas, 'Color', 'b', 'LineWidth',1.5); % Should also plot standard deviation
    hold on;
    plot(f_post,mas_post, 'Color', 'r', 'LineWidth',1.5);
    hold on;
    % fill([f'; flip(f')], [mas' - mas_pre_std'/sqrt(64); flip(mas' + mas_pre_std'/sqrt(64))], 'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold on;
    % fill([f'; flip(f')], [mas_post' - mas_post_std'/sqrt(64); flip(mas_post' + mas_post_std'/sqrt(64))], 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
end
legend('Pre-PL', 'Post-PL');
average_across_subjects = mean(all_subject_spectra_pre_PL, 1);
average_across_subjects_post = mean(all_subject_spectra_post_PL, 1);

%%
figure;   
% LTG
load('patient_data_LTG.mat'); % data is not provided for ethical reasons
for ii = 1:14   
    subplot(3,5,ii);
    ii
    D = Data_LTG(ii).pre.DataOut;
    average_spectrum = [];
    average_spectrum_post = [];
    for channel_n = 1:64 
        channel_n
    for i = 1:length(D.trial) % Iterate through trials and channels, for each individual subject  
            x = D.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            baseline = mean(D.trial{i},1);
            x = x - baseline;
            N = length(x);
            f_sampling = 1000; 
            f = (0:N-1) * f_sampling / N;
            X = fft(x);
            Pxx = (1/(N*f_sampling)) * abs(X).^2;
            indx=find(f>=2 & f<=45);
            Pxx =Pxx(indx);
            f = f(indx);
            Pxx = Pxx/trapz(f,Pxx);
            average_spectrum = [average_spectrum; Pxx];        
    end
    mas = mean(average_spectrum, 1); % mean for trial and channel    
    % Post-drug
    D_post = Data_LTG(ii).post.DataOut;
    for i = 1:length(D_post.trial) % Iterate through trials and channels, for each individual subject  
            x_post = D_post.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            % Subtract mean for each trial
            baseline_post = mean(D_post.trial{i},1);
            x_post = x_post - baseline_post;
            N_post = length(x_post);
            f_sampling = 1000; 
            f_post = (0:N-1) * f_sampling / N_post;
            X_post = fft(x_post);
            Pxx_post = (1/(N_post*f_sampling)) * abs(X_post).^2;
            
            indx_post=find(f_post>=2 & f_post<=45);
            Pxx_post =Pxx_post(indx_post);
            f_post = f_post(indx_post);
            Pxx_post = Pxx_post/trapz(f_post,Pxx_post);
            average_spectrum_post = [average_spectrum_post; Pxx_post];       
    end
    mas_post = mean(average_spectrum_post, 1); % mean for trial and channel
    end
    all_subject_spectra_pre_LTG(ii, :) = mas;
    all_subject_spectra_post_LTG(ii, :) = mas_post;
    mas_pre_std = std(mas, 0, 2);
    mas_post_std = std(mas_post, 0, 2); 
    plot(f,mas, 'Color', 'b', 'LineWidth',1.5); % Should also plot standard deviation
    hold on;
    plot(f_post,mas_post, 'Color', 'r', 'LineWidth',1.5);
    hold on;
    fill([f'; flip(f')], [mas' - mas_pre_std'/sqrt(64); flip(mas' + mas_pre_std'/sqrt(64))], 'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold on;
    fill([f'; flip(f')], [mas_post' - mas_post_std'/sqrt(64); flip(mas_post' + mas_post_std'/sqrt(64))], 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
end
legend('Pre-LTG', 'Post-LTG');
average_across_subjects = mean(all_subject_spectra_pre_LTG, 1);
average_across_subjects_post = mean(all_subject_spectra_post_LTG, 1);



figure;
% LEV
load('patient_data_LEV.mat'); % data is not provided for ethical reasons
for ii = 1:14    
    ii
    subplot(3,5,ii);
    D = Data_LEV(ii).pre.DataOut;
    average_spectrum = [];
    average_spectrum_post = [];
    for channel_n = 1:64 
        channel_n
    for i = 1:length(D.trial) % Iterate through trials and channels, for each individual subject  
            x = D.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            baseline = mean(D.trial{i},1);
            x = x - baseline;
            N = length(x);
            f_sampling = 1000; 
            f = (0:N-1) * f_sampling / N;
            X = fft(x);
            Pxx = (1/(N*f_sampling)) * abs(X).^2;
            indx=find(f>=2 & f<=45);
            Pxx =Pxx(indx);
            f = f(indx);
            Pxx = Pxx/trapz(f,Pxx);
            average_spectrum = [average_spectrum; Pxx];        
    end
    mas = mean(average_spectrum, 1); % mean for trial and channel 
    % Post-drug
    D_post = Data_LEV(ii).post.DataOut;
    for i = 1:length(D_post.trial) % Iterate through trials and channels, for each individual subject  
            x_post = D_post.trial{i}(channel_n, :); % Single trial (segment of EEG data)
            % Subtract mean for each trial
            baseline_post = mean(D_post.trial{i},1);
            x_post = x_post - baseline_post;
            N_post = length(x_post);
            f_sampling = 1000; 
            f_post = (0:N-1) * f_sampling / N_post;
            X_post = fft(x_post);
            Pxx_post = (1/(N_post*f_sampling)) * abs(X_post).^2;
            
            indx_post=find(f_post>=2 & f_post<=45);
            Pxx_post =Pxx_post(indx_post);
            f_post = f_post(indx_post);
            Pxx_post = Pxx_post/trapz(f_post,Pxx_post);
            average_spectrum_post = [average_spectrum_post; Pxx_post];       
    end
    mas_post = mean(average_spectrum_post, 1); % mean for trial and channel
    end
    all_subject_spectra_pre_LEV(ii, :) = mas;
    all_subject_spectra_post_LEV(ii, :) = mas_post;
    mas_pre_std = std(mas, 0, 2);
    mas_post_std = std(mas_post, 0, 2); 
    plot(f,mas, 'Color', 'b', 'LineWidth',1.5); % Should also plot standard deviation
    hold on;
    plot(f_post,mas_post, 'Color', 'r', 'LineWidth',1.5);
    hold on;
    fill([f'; flip(f')], [mas' - mas_pre_std'/sqrt(64); flip(mas' + mas_pre_std'/sqrt(64))], 'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold on;
    fill([f'; flip(f')], [mas_post' - mas_post_std'/sqrt(64); flip(mas_post' + mas_post_std'/sqrt(64))], 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
end
legend('Pre-LEV', 'Post-LEV');
average_across_subjects = mean(all_subject_spectra_pre_LEV, 1);
average_across_subjects_post = mean(all_subject_spectra_post_LEV, 1);


    %%

figure;
sigma = 2;  % Standard deviation of the Gaussian
plot(f(1:57), log(movmean(mean(all_subject_spectra_pre_PL(:,1:57),1),5)));
hold on;
plot(f(1:57), log(movmean(mean(all_subject_spectra_post_PL(:,1:57),1),5)));

% figure;
% hold on;
% plot(data_psd_pre_PL);
% hold on;
% plot(data_psd_post_PL);





