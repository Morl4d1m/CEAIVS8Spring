[x, fs] = audioread('backgroundMeasurement_1008.wav');
x = x - mean(x);

SPL_raw = 20*log10(rms(x) / 20e-6);
fprintf('Raw SPL (no calibration): %.2f dB SPL\n', SPL_raw);
