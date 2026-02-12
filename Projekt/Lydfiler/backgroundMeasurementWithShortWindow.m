%% ISO 3382-2 Background Noise Measurement
% Background noise evaluation using calibrated microphones
% Author: Christian Lykke
% Date: 30-01-2026

clear; clc;

%% Flags
plotPerMic    = 0;   % true = plot A vs Z for each mic individually
plotAverage   = 0;   % true = plot only spatial average
plotWindowed  = 0;   % true = plot short-window LAeq for comparison with SPL meter
window_s      = 10;     % window length in seconds for short-window LAeq

%% Paths
basePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
calFile  = fullfile(basePath, 'kalibrering30012026.mat');

%% Load calibration data (for verification)
S = load(calFile);

if isfield(S,'K_mic')
    K_mic = S.K_mic;
elseif isfield(S,'K')
    K_mic = S.K;
else
    error('Calibration file does not contain K_mic or K');
end

if numel(K_mic) ~= 8
    error('Calibration vector must contain 8 microphone constants');
end

%% Time window (seconds)
t_start = 2*60 + 30;   % 2:30
t_end   = 2*60 + 40;%10*60 + 30;  % 10:30

%% Preallocate
N_mics  = 8;
LAeq_A = zeros(1,N_mics);
SPL_Z  = zeros(1,N_mics);
pA_all = cell(1,N_mics);
pZ_all = cell(1,N_mics);
fs_all = zeros(1,N_mics);

fprintf('\nISO 3382-2 BACKGROUND NOISE MEASUREMENT\n');
fprintf('--------------------------------------\n');
fprintf('Time window: %02d:%02d to %02d:%02d (%.0f s)\n', ...
        floor(t_start/60), mod(t_start,60), ...
        floor(t_end/60), mod(t_end,60), ...
        t_end - t_start);
fprintf('Reference  : 20 µPa\n\n');

%% Loop over microphones
for mic = 1:N_mics
    % Load audio
    fileName = sprintf('backgroundMeasurement_100%d.wav', mic);
    filePath = fullfile(basePath, fileName);
    [x, fs] = audioread(filePath);
    x = x - mean(x);   % DC removal

    % Extract time segment
    idx_start = round(t_start * fs) + 1;
    idx_end   = round(t_end * fs);
    if idx_end > length(x)
        error('File %s is shorter than 10:30', fileName);
    end
    x = x(idx_start:idx_end);

    % Already in Pa; no K_mic multiplication
    p = x;

    % Z-weighted (unweighted)
    SPL_Z(mic) = 20*log10(rms(p)/20e-6);
    pZ_all{mic} = p;
    fs_all(mic) = fs;

    % A-weighted
    A = weightingFilter('A-weighting','SampleRate',fs);
    pA = A(p);
    LAeq_A(mic) = 10*log10(mean(pA.^2)/(20e-6)^2);
    pA_all{mic} = pA;

    % Display results
    fprintf('Mic %d: Z-weighted = %5.2f dB, A-weighted = %5.2f dB(A)\n', ...
        mic, SPL_Z(mic), LAeq_A(mic));
end

%% Spatial averages
LAeq_A_mean = 10*log10(mean(10.^(LAeq_A/10)));
SPL_Z_mean  = 10*log10(mean(10.^(SPL_Z/10)));

fprintf('\n--------------------------------------\n');
fprintf('Spatial average Z-weighted SPL : %5.2f dB\n', SPL_Z_mean);
fprintf('Spatial average A-weighted LAeq: %5.2f dB(A)\n', LAeq_A_mean);
fprintf('--------------------------------------\n\n');

%% Verification snippet: 1 kHz calibration vs background
fprintf('VERIFICATION OF CALIBRATION AND BACKGROUND\n');
calFiles = arrayfun(@(i) sprintf('kalibrering30012026_enkeltfiler_%04d.wav',1000+i), 1:N_mics, 'UniformOutput',false);
for i = 1:N_mics
    [x, fs] = audioread(fullfile(basePath, calFiles{i}));
    x = x - mean(x);
    
    % Narrowband 1 kHz RMS
    bp = designfilt('bandpassiir', 'FilterOrder',6, ...
                    'HalfPowerFrequency1',950, 'HalfPowerFrequency2',1050, ...
                    'SampleRate', fs);
    x_f = filtfilt(bp, x);
    SPL_cal = 20*log10(rms(x_f)/20e-6); % Z-weighted
    
    % Background RMS
    SPL_bg_Z = 20*log10(rms(pZ_all{i})/20e-6);
    SPL_bg_A = 10*log10(mean(pA_all{i}.^2)/(20e-6)^2);
    
    fprintf('Mic %d 1 kHz calibrator: %.2f dB SPL | Background: Z = %.2f dB, A = %.2f dB(A)\n', ...
        i, SPL_cal, SPL_bg_Z, SPL_bg_A);
end
fprintf('--------------------------------------\n\n');

%% Short-window LAeq (time-resolved) for SPL meter comparison
if plotWindowed
    fprintf('Computing short-window LAeq (%d s) for all microphones...\n', window_s);
    for mic = 1:N_mics
        N_win = round(window_s * fs_all(mic));
        num_win = floor(length(pA_all{mic})/N_win);
        LAeq_win = zeros(1,num_win);
        t_win = (0:num_win-1)*window_s;
        for w = 1:num_win
            idx = (1:N_win) + (w-1)*N_win;
            LAeq_win(w) = 10*log10(mean(pA_all{mic}(idx).^2)/(20e-6)^2);
        end
        if plotWindowed
            figure('Name',sprintf('Mic %d short-window LAeq',mic),'NumberTitle','off');
            plot(t_win,LAeq_win,'r','LineWidth',1.2);
            xlabel('Time [s]'); ylabel('LAeq [dB(A)]');
            title(sprintf('Mic %d short-window LAeq (%d s windows)', mic, window_s));
            grid on;
        end
    end
end

%% Plotting options
if plotPerMic
    figure('Name','Background Noise A vs Z','NumberTitle','off');
    for mic = 1:N_mics
        t_vec = (0:length(pA_all{mic})-1)/fs_all(mic);
        subplot(4,2,mic)
        plot(t_vec,20*log10(abs(pZ_all{mic})/20e-6),'b'); hold on;
        plot(t_vec,20*log10(abs(pA_all{mic})/20e-6),'r');
        xlabel('Time [s]'); ylabel('SPL [dB]');
        title(sprintf('Mic %d',mic));
        legend('Z-weighted','A-weighted');
        grid on;
    end
end

if plotAverage
    figure('Name','Spatial Average Background Noise','NumberTitle','off');
    t_vec = (0:length(pA_all{1})-1)/fs_all(1);
    pZ_avg = mean(cell2mat(cellfun(@(c) c(:), pZ_all,'UniformOutput',false)),2);
    pA_avg = mean(cell2mat(cellfun(@(c) c(:), pA_all,'UniformOutput',false)),2);
    plot(t_vec,20*log10(abs(pZ_avg)/20e-6),'b'); hold on;
    plot(t_vec,20*log10(abs(pA_avg)/20e-6),'r');
    xlabel('Time [s]'); ylabel('SPL [dB]');
    title('Spatial average of 8 microphones');
    legend('Z-weighted','A-weighted');
    grid on;
end
