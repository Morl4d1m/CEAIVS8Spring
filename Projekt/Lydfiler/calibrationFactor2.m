
%% ================= USER SETTINGS ==========================
folder = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
filePattern = 'kalibrering20022026__1%03d.wav';

N = 8;

% Calibration tone definition
calFreq = 1000;          % Hz
calLevel_dB = 94;        % dB SPL
p_ref = 20e-6;           % Pa

% Method selection:
%   "hardmath_rms"     = RMS of bandpassed 1 kHz tone (recommended)
%   "hardmath_sinefit" = sine fit at 1 kHz (very robust)
%   "matlab_builtin"   = uses calibrateMicrophone() (if available)
calMethod = "hardmath_sinefit";

% Bandpass used for hard-math methods
bpLow  = 900;            % Hz
bpHigh = 1100;           % Hz

% Segment selection (to avoid the fit-on/fit-off clipping)
% The script auto-finds the stable region, but these parameters control it.
segmentDuration_s = 2.0;     % seconds used for calibration
searchMargin_s = 0.5;        % ignore first/last part of file

% Debug plots
plotPerMic = 1;          % 1 = show plots per mic
plotSummary = 1;         % 1 = show summary bar plot

% Output MAT file
outMatFile = fullfile(folder, 'micCalibrationConstants_94dB_1kHz_20022026.mat');
%% ==========================================================


fprintf('\nMICROPHONE CALIBRATION\n');
fprintf('----------------------\n');
fprintf('Folder     : %s\n', folder);
fprintf('Pattern    : %s\n', filePattern);
fprintf('Method     : %s\n', calMethod);
fprintf('Tone       : %.0f Hz @ %.1f dB SPL\n\n', calFreq, calLevel_dB);

%% Reference pressure for 94 dB SPL
p_rms_ref = p_ref * 10^(calLevel_dB/20);   % RMS pressure in Pa

fprintf('Reference RMS pressure at %.1f dB SPL = %.6f Pa\n\n', calLevel_dB, p_rms_ref);

%% Preallocate
K_mic = zeros(1,N);   % Pa/FS
cal = struct();
cal.mic = repmat(struct(),1,N);

%% Loop microphones
for mic = 1:N

    fileName = sprintf(filePattern, mic);
    filePath = fullfile(folder, fileName);

    if ~exist(filePath,'file')
        error('File not found: %s', filePath);
    end

    [x, fs] = audioread(filePath);
    x = x(:);
    x = x - mean(x);   % remove DC

    t = (0:length(x)-1)/fs;

    % Ignore first/last margin
    idx0 = round(searchMargin_s * fs) + 1;
    idx1 = length(x) - round(searchMargin_s * fs);

    if idx1 <= idx0
        error('File too short or margin too large for mic %d', mic);
    end

    x_use = x(idx0:idx1);
    t_use = t(idx0:idx1);

    % Bandpass around 1 kHz for robust detection
    bp = designfilt('bandpassiir', ...
        'FilterOrder', 6, ...
        'HalfPowerFrequency1', bpLow, ...
        'HalfPowerFrequency2', bpHigh, ...
        'SampleRate', fs);

    x_bp = filtfilt(bp, x_use);

    % Find a stable region:
    % We compute a short-time RMS and pick the strongest, most stable segment.
    win_s = 0.050;                      % 50 ms RMS window
    hop_s = 0.010;                      % 10 ms hop
    win = max(16, round(win_s*fs));
    hop = max(1, round(hop_s*fs));

    nFrames = floor((length(x_bp)-win)/hop) + 1;
    if nFrames < 10
        error('Not enough data to analyze stability for mic %d', mic);
    end

    rmsFrames = zeros(nFrames,1);
    for k = 1:nFrames
        i1 = (k-1)*hop + 1;
        i2 = i1 + win - 1;
        seg = x_bp(i1:i2);
        rmsFrames(k) = rms(seg);
    end

    % Find the best continuous region of segmentDuration_s
    segLen = round(segmentDuration_s * fs);
    segFrames = floor((segLen - win)/hop) + 1;
    if segFrames < 5
        error('segmentDuration_s too small for mic %d', mic);
    end

    bestScore = -Inf;
    bestStartFrame = 1;

    for k = 1:(nFrames - segFrames)
        r = rmsFrames(k:k+segFrames-1);

        % Score = high RMS but low variation
        score = mean(r) / (std(r) + 1e-12);

        if score > bestScore
            bestScore = score;
            bestStartFrame = k;
        end
    end

    % Convert frame index to sample indices in x_use
    segStart = (bestStartFrame-1)*hop + 1;
    segEnd   = segStart + segLen - 1;

    if segEnd > length(x_use)
        segEnd = length(x_use);
        segStart = segEnd - segLen + 1;
    end

    x_seg = x_use(segStart:segEnd);
    t_seg = t_use(segStart:segEnd);

    %% ================= CALIBRATION METHODS =====================

    switch calMethod

        case "hardmath_rms"
            % Bandpass and RMS
            x_seg_bp = filtfilt(bp, x_seg);
            x_rms = rms(x_seg_bp);

            % Calibration constant: Pa/FS
            K = p_rms_ref / x_rms;

        case "hardmath_sinefit"
            % Sine-fit at exactly 1 kHz:
            % x(t) = A*sin(wt) + B*cos(wt)
            % amplitude_peak = sqrt(A^2 + B^2)
            % amplitude_rms  = amplitude_peak / sqrt(2)

            w = 2*pi*calFreq;
            s = sin(w*t_seg(:));
            c = cos(w*t_seg(:));

            M = [s c];
            theta = M \ x_seg(:);    % least squares
            A = theta(1);
            B = theta(2);

            amp_peak = sqrt(A^2 + B^2);
            x_rms = amp_peak / sqrt(2);

            K = p_rms_ref / x_rms;

        case "matlab_builtin"
            % Uses MATLAB function calibrateMicrophone()
            % This requires Audio Toolbox (and sometimes DSP System Toolbox).
            %
            % WARNING:
            % calibrateMicrophone returns sensitivity in V/Pa depending on input type.
            % For WAV files (normalized -1..1), we need Pa/FS.
            %
            % So we still compute K as p_rms_ref / x_rms, but we use
            % calibrateMicrophone() only to estimate x_rms robustly.

            try
                % calibrateMicrophone expects audio in Pascal or Volt depending on mode.
                % For normalized wav, it will not directly return Pa/FS.
                % So we use it only as a helper (if it exists).
                [~, x_rms] = calibrateMicrophone(x_seg, fs, ...
                    "CalibratorLevel", calLevel_dB, ...
                    "CalibratorFrequency", calFreq);

                % If x_rms is empty or nonsense, fallback
                if isempty(x_rms) || ~isfinite(x_rms) || x_rms <= 0
                    error('calibrateMicrophone returned invalid RMS');
                end

                K = p_rms_ref / x_rms;

            catch
                warning('Mic %d: calibrateMicrophone() failed, falling back to sinefit.', mic);

                w = 2*pi*calFreq;
                s = sin(w*t_seg(:));
                c = cos(w*t_seg(:));
                theta = [s c] \ x_seg(:);
                amp_peak = sqrt(theta(1)^2 + theta(2)^2);
                x_rms = amp_peak / sqrt(2);
                K = p_rms_ref / x_rms;
            end

        otherwise
            error('Unknown calMethod: %s', calMethod);
    end

    %% ============================================================

    % Store
    K_mic(mic) = K;

    % Diagnostic SPL if this K were applied (should be 94 dB)
    p_est = x_seg * K;
    SPL_est = 20*log10(rms(p_est)/p_ref);

    cal.mic(mic).file = fileName;
    cal.mic(mic).fs = fs;
    cal.mic(mic).K_PaPerFS = K;
    cal.mic(mic).segmentStart_s = t_seg(1);
    cal.mic(mic).segmentEnd_s = t_seg(end);
    cal.mic(mic).x_rms_FS = rms(x_seg);
    cal.mic(mic).SPL_est_dB = SPL_est;
    cal.mic(mic).score = bestScore;

    fprintf('Mic %d: K = %.6f Pa/FS  |  SPL check = %.2f dB\n', mic, K, SPL_est);

    %% Plot
    if plotPerMic
        figure('Name',sprintf('Calibration Mic %d',mic),'NumberTitle','off');

        subplot(3,1,1)
        plot(t, x, 'k');
        grid on;
        xlabel('Time [s]');
        ylabel('Amplitude [FS]');
        title(sprintf('Mic %d raw (full file)', mic));

        subplot(3,1,2)
        plot(t_use, x_bp, 'b'); hold on;
        xline(t_seg(1),'r','LineWidth',1.5);
        xline(t_seg(end),'r','LineWidth',1.5);
        grid on;
        xlabel('Time [s]');
        ylabel('Bandpassed (FS)');
        title('Bandpassed (900-1100 Hz) + chosen segment');

        subplot(3,1,3)
        plot(t_seg, x_seg, 'k'); grid on;
        xlabel('Time [s]');
        ylabel('Amplitude [FS]');
        title(sprintf('Chosen segment (%.2f s) | SPL check %.2f dB', ...
            segmentDuration_s, SPL_est));
    end

end

%% Save MAT file
cal.info.method = calMethod;
cal.info.calFreq = calFreq;
cal.info.calLevel_dB = calLevel_dB;
cal.info.p_ref = p_ref;
cal.info.p_rms_ref = p_rms_ref;
cal.info.bpLow = bpLow;
cal.info.bpHigh = bpHigh;
cal.info.segmentDuration_s = segmentDuration_s;

save(outMatFile, 'K_mic', 'cal');

fprintf('\nSaved calibration file:\n%s\n\n', outMatFile);

%% Summary plot
if plotSummary
    figure('Name','Calibration Constants (Pa/FS)','NumberTitle','off');
    bar(1:N, K_mic);
    grid on;
    xlabel('Microphone index');
    ylabel('K [Pa/FS]');
    title(sprintf('Calibration constants (%s)', calMethod));
end
