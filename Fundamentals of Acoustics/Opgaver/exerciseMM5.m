%% DOA estimation using two microphones
% Mic spacing: 0.4 m
% Sampling frequency: 48 kHz
% Far-field assumption

%% Parameters
d = 0.4;          % microphone spacing (m)
c = 343;          % speed of sound (m/s)

%% Load signals
[left, fs]  = audioread('left_mic.wav');
[right, fs] = audioread('right_mic.wav');

left  = left(:,1);
right = right(:,1);

t = (0:length(left)-1)/fs;

fprintf('Sampling frequency: %d Hz\n', fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (a) Cross-correlation DOA estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[corr_lr, lags] = xcorr(left, right);

[~, idx] = max(abs(corr_lr));
lag_samples = lags(idx);
tau = lag_samples / fs;

theta_xcorr = asin((c * tau) / d);
theta_xcorr_deg = rad2deg(theta_xcorr);

fprintf('\n--- Cross-correlation result ---\n');
fprintf('Lag (samples): %d\n', lag_samples);
fprintf('Time delay (s): %.6f\n', tau);
fprintf('Angle (deg): %.2f\n', theta_xcorr_deg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (b) Delay-and-Sum Beamforming
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

angles = -90:0.1:90;
power_out = zeros(size(angles));

for k = 1:length(angles)
    
    theta = deg2rad(angles(k));
    tau_k = (d * sin(theta)) / c;
    delay_samples = tau_k * fs;
    
    right_delayed = delayseq(right, delay_samples);
    y = left + right_delayed;
    
    power_out(k) = sum(y.^2);
end

[~, idx_bf] = max(power_out);
theta_bf_deg = angles(idx_bf);

fprintf('\n--- Delay-and-Sum result ---\n');
fprintf('Angle (deg): %.2f\n', theta_bf_deg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (c) Plot signals at estimated beamformer angle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta_best = deg2rad(theta_bf_deg);
tau_best = (d * sin(theta_best)) / c;
delay_best = tau_best * fs;

right_aligned = delayseq(right, delay_best);
y_best = left + right_aligned;

figure;
plot(t, left, 'b'); hold on;
plot(t, right, 'r');
plot(t, y_best, 'k', 'LineWidth', 1.2);
legend('Left mic','Right mic','Beamformer output');
xlabel('Time (s)');
ylabel('Amplitude');
title('Signals and Beamformer Output');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (d) Upsample to 192 kHz and repeat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

left_up  = resample(left, 4, 1);
right_up = resample(right, 4, 1);

fs_up = fs * 4;
t_up = (0:length(left_up)-1)/fs_up;

%% Cross-correlation (upsampled)
[corr_up, lags_up] = xcorr(left_up, right_up);

[~, idx_up] = max(abs(corr_up));
lag_up = lags_up(idx_up);
tau_up = lag_up / fs_up;

theta_xcorr_up = asin((c * tau_up) / d);
theta_xcorr_up_deg = rad2deg(theta_xcorr_up);

fprintf('\n--- Cross-correlation (192 kHz) ---\n');
fprintf('Angle (deg): %.2f\n', theta_xcorr_up_deg);

%% Beamforming (upsampled)
angles_up = -90:0.1:90;
power_up = zeros(size(angles_up));

for k = 1:length(angles_up)
    
    theta = deg2rad(angles_up(k));
    tau_k = (d * sin(theta)) / c;
    delay_samples = tau_k * fs_up;
    
    right_delayed = delayseq(right_up, delay_samples);
    y = left_up + right_delayed;
    
    power_up(k) = sum(y.^2);
end

[~, idx_bf_up] = max(power_up);
theta_bf_up_deg = angles_up(idx_bf_up);

fprintf('\n--- Delay-and-Sum (192 kHz) ---\n');
fprintf('Angle (deg): %.2f\n', theta_bf_up_deg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (e) Compare with true angle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = 1.5;
y = 4.5;

theta_true = atan2(x, y);
theta_true_deg = rad2deg(theta_true);

fprintf('\n--- True angle ---\n');
fprintf('True angle (deg): %.2f\n', theta_true_deg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Beam pattern plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(angles, power_out/max(power_out), 'LineWidth', 1.5);
xlabel('Angle (deg)');
ylabel('Normalized Power');
title('Beamformer Spatial Response (48 kHz)');
grid on;

figure;
plot(angles_up, power_up/max(power_up), 'LineWidth', 1.5);
xlabel('Angle (deg)');
ylabel('Normalized Power');
title('Beamformer Spatial Response (192 kHz)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%