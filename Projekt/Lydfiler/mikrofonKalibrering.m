% Nuværende fil er målt med all gain på 50. på micstasy
% Folder and filename pattern
folder = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
filePattern = 'kalibrering11022026Lab2_%04d.wav';

N = 1;                     % number of files
K_mic = zeros(1, N);
A_rms_raw = zeros(1, N);
p_rms = zeros(1, N);
SPL = zeros(1, N);

fprintf('Calibration results:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Microphone\tA_rms (raw)\tK (Pa/unit)\tp_rms (Pa)\tSPL (dB)\n');
fprintf('-------------------------------------------------------------\n');

for i = 1:N
    % Read audio file
    filename = fullfile(folder, sprintf(filePattern, 1000 + i));
    [x, fs] = audioread(filename); % Saves as raw ADC output

    % Remove DC
    x = x - mean(x);

    % Narrow band around 1 kHz
    bp = designfilt('bandpassiir', ...
        'FilterOrder', 6, ...
        'HalfPowerFrequency1', 950, ...
        'HalfPowerFrequency2', 1050, ...
        'SampleRate', fs);

    x_f = filtfilt(bp, x); % Actually does the filtering, both forwards and backwards

    % RMS amplitude (raw digital value)
    A_rms_raw(i) = rms(x_f);
    % This should be the same as the system gain

    % Calibration factor (Pa per digital unit)
    % The 1 is assumed from the calibrator outputting 1 Pa RMS
    % Can also be found as: p_cal = 20e-6 * 10^(94/20);
    K_mic(i) = 1.0 / A_rms_raw(i);

    % Adjusted (physical) values
    p_rms(i) = rms(K_mic(i) * x_f);
    SPL(i) = 20 * log10(p_rms(i) / 20e-6);

    % Print values
    fprintf('%d\t\t\t%.4e\t%.4e\t%.4e\t%.2f\n', ...
        i, A_rms_raw(i), K_mic(i), p_rms(i), SPL(i));
end

fprintf('-------------------------------------------------------------\n');
save('kalibrering11022026Lab2.mat', 'K_mic', 'A_rms_raw', 'SPL');
fprintf('Saved data to .mat file\n');
fprintf('-------------------------------------------------------------\n');