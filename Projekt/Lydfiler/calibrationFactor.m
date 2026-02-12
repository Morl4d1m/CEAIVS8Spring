
%% USER SETTINGS
folder = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
filePattern = 'kalibrering020220262__1%03d.wav';

numMics = 8;
toneFreq = 1000;
calLevel_dBSPL = 94;
pref = 20e-6;
targetPa_rms = pref * 10^(calLevel_dBSPL/20);   % 1.002374 Pa RMS

% Bandpass width for tone isolation
bpHalfWidthHz = 30;   % +-30 Hz around 1 kHz

% Envelope settings (for finding steady region)
rmsWindow_ms = 50;    % RMS envelope window length
stableThreshold_dB = 0.5;  % plateau must stay within +-0.5 dB

% Reject clipped samples (full scale)
clipThreshold = 0.999;

% Optional: only if you know ADC full-scale peak volts for sample=1.0
FS_Vpeak = NaN;

outputMatFile = fullfile(folder, 'micCalibrationConstants_94dB_1kHz.mat');

%% STORAGE
cal = struct();
cal.folder = folder;
cal.filePattern = filePattern;
cal.numMics = numMics;
cal.toneFreq = toneFreq;
cal.calLevel_dBSPL = calLevel_dBSPL;
cal.targetPa_rms = targetPa_rms;
cal.method = "auto-steady-region bandpass RMS";
cal.bpHalfWidthHz = bpHalfWidthHz;
cal.rmsWindow_ms = rmsWindow_ms;
cal.stableThreshold_dB = stableThreshold_dB;
cal.clipThreshold = clipThreshold;
cal.FS_Vpeak = FS_Vpeak;

cal.mic = repmat(struct( ...
    'file', "", ...
    'Fs', NaN, ...
    'rmsFS', NaN, ...
    'calFactor_PaPerFS', NaN, ...
    'calFactor_PaPerVolt', NaN, ...
    'SPLcheck', NaN, ...
    'steadyStart_s', NaN, ...
    'steadyEnd_s', NaN, ...
    'clipFraction', NaN ...
), numMics, 1);

%% MAIN LOOP
for m = 1:numMics
    fprintf("------------------------------------------------------------\n");
    fprintf("Mic %d / %d\n", m, numMics);

    file = fullfile(folder, sprintf(filePattern, m));
    if ~isfile(file)
        error("Missing file: %s", file);
    end

    [x, Fs] = audioread(file);
    x = double(x);

    if size(x,2) > 1
        x = x(:,1);
    end

    x = x - mean(x);
    t = (0:length(x)-1)/Fs;

    % Clip detection
    clipFraction = mean(abs(x) > clipThreshold);

    % Bandpass around 1 kHz
    f1 = max(10, toneFreq - bpHalfWidthHz);
    f2 = min(Fs/2 - 10, toneFreq + bpHalfWidthHz);

    bp = designfilt('bandpassiir', ...
        'FilterOrder', 6, ...
        'HalfPowerFrequency1', f1, ...
        'HalfPowerFrequency2', f2, ...
        'SampleRate', Fs);

    xTone = filtfilt(bp, x);

    % Compute short-time RMS envelope
    win = max(16, round((rmsWindow_ms/1000) * Fs));
    env = sqrt(movmean(xTone.^2, win));

    % Avoid log(0)
    env_dB = 20*log10(env + 1e-20);

    % Find plateau region:
    % Use top 80% of envelope level as "tone active"
    peakEnv = max(env);
    active = env > (0.8 * peakEnv);

    % Find longest contiguous active region
    idx = find(active);
    if isempty(idx)
        error("Mic %d: Could not detect tone region.", m);
    end

    breaks = [1; find(diff(idx) > 1) + 1; length(idx)+1];
    segLens = diff(breaks);
    [~, kBest] = max(segLens);
    seg = idx(breaks(kBest) : breaks(kBest+1)-1);

    % Now inside that segment, find stable part:
    segEnv_dB = env_dB(seg);
    segMedian = median(segEnv_dB);

    stable = abs(segEnv_dB - segMedian) < stableThreshold_dB;
    stableIdx = seg(stable);

    if length(stableIdx) < 0.2*length(seg)
        warning("Mic %d: Stable region is small. Using center 50%% of active region.", m);
        stableIdx = seg(round(0.25*length(seg)) : round(0.75*length(seg)));
    end

    % Compute RMS ONLY on stable region
    xStable = xTone(stableIdx);
    rmsFS = sqrt(mean(xStable.^2));

    % Calibration factor
    PaPerFS = targetPa_rms / rmsFS;

    % SPL check
    p_rms_est = rmsFS * PaPerFS;
    SPLcheck = 20*log10(p_rms_est / pref);

    % Optional Pa/V
    if ~isnan(FS_Vpeak)
        Vrms = rmsFS * FS_Vpeak;
        PaPerVolt = targetPa_rms / Vrms;
    else
        PaPerVolt = NaN;
    end

    % Store
    cal.mic(m).file = file;
    cal.mic(m).Fs = Fs;
    cal.mic(m).rmsFS = rmsFS;
    cal.mic(m).calFactor_PaPerFS = PaPerFS;
    cal.mic(m).calFactor_PaPerVolt = PaPerVolt;
    cal.mic(m).SPLcheck = SPLcheck;
    cal.mic(m).steadyStart_s = t(stableIdx(1));
    cal.mic(m).steadyEnd_s = t(stableIdx(end));
    cal.mic(m).clipFraction = clipFraction;

    fprintf("rmsFS (steady): %.8g\n", rmsFS);
    fprintf("Pa/FS: %.8g\n", PaPerFS);
    fprintf("SPL check: %.4f dB\n", SPLcheck);
    fprintf("Steady region: %.3f s to %.3f s\n", ...
        cal.mic(m).steadyStart_s, cal.mic(m).steadyEnd_s);
    fprintf("Clip fraction: %.6f\n", clipFraction);
end

%% SUMMARY
fprintf("\n============================================================\n");
fprintf("SUMMARY\n");
for m = 1:numMics
    fprintf("Mic %d: rmsFS=%.6g | Pa/FS=%.6g | steady=%.2f-%.2f s | clip=%.4f\n", ...
        m, cal.mic(m).rmsFS, cal.mic(m).calFactor_PaPerFS, ...
        cal.mic(m).steadyStart_s, cal.mic(m).steadyEnd_s, ...
        cal.mic(m).clipFraction);
end
fprintf("============================================================\n");

%% SANITY CHECK
r = [cal.mic.rmsFS];
ratio = r / median(r);
ratio_dB = 20*log10(ratio);

fprintf("\nRelative tone levels vs median:\n");
for m = 1:numMics
    fprintf("Mic %d: %+6.2f dB\n", m, ratio_dB(m));
end

if any(abs(ratio_dB) > 3)
    fprintf(2, "\nWARNING: One or more microphones differ by > 3 dB.\n");
    fprintf(2, "That is large for identical mics in the same calibrator.\n");
    fprintf(2, "Most likely: tone not fully sealed on some mics, or leak.\n");
end

%% SAVE
save(outputMatFile, "cal");
fprintf("\nSaved calibration MAT file:\n%s\n", outputMatFile);
