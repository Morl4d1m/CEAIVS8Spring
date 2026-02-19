%% ISO 3382-2 Background Noise Measurement
% Optimized + ISO compliant tables
% MATLAB R2023a compatible
%
% Author: Christian Lykke
% Updated: Optimized version


%% =========================== FLAGS ===========================
doZ = true;
doA = true;
doC = true;

doOctave      = 0;
doThirdOctave = 0;

plotFFTAverage = 0;
plotOctaveBars = 0;
plotThirdOctaveBars = 0;

%% =========================== PATHS ===========================
basePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
calFile  = fullfile(basePath,'micCalibrationConstants_94dB_1kHz_16022026.mat');

N = 8;
p_ref = 20e-6;

%% ==================== LOAD CALIBRATION =======================
S = load(calFile);

if isfield(S,'K_mic')
    K_mic = S.K_mic;
elseif isfield(S,'K')
    K_mic = S.K;
elseif isfield(S,'cal')
    K_mic = [S.cal.mic.K_PaPerFS];
else
    error('Calibration constants not found.');
end

%% ======================= TIME WINDOW =========================
t_start =60;% 2*60 + 30;
t_end   =length(x)/fs;;% 10*60 + 30;

%% ================= PRE-READ SAMPLERATE =======================
fs=44100;

%% ================= CREATE FILTERS (ONCE) =====================
if doA
    Afilter = weightingFilter('A-weighting','SampleRate',fs);
end
if doC
    Cfilter = weightingFilter('C-weighting','SampleRate',fs);
end

if doOctave
    FB_oct = octaveFilterBank( ...
        'SampleRate',fs, ...
        'Bandwidth','1 octave', ...
        'FrequencyRange',[31.5 20000]);
    fc_oct = getCenterFrequencies(FB_oct);
end

if doThirdOctave
    FB_third = octaveFilterBank( ...
        'SampleRate',fs, ...
        'Bandwidth','1/3 octave', ...
        'FrequencyRange',[10 20000]);
    fc_third = getCenterFrequencies(FB_third);
end

%% ===================== PREALLOCATE ===========================
LAeq_Z = zeros(1,N);
LAeq_A = zeros(1,N);
LAeq_C = zeros(1,N);

p_all = cell(1,N);
pA_all = cell(1,N);
pC_all = cell(1,N);

%% ====================== MAIN LOOP ============================
for mic = 1:N

    fileName = sprintf('backgroundMeasurement16022026__100%d.wav',mic);
    [x, ~] = audioread(fullfile(basePath,fileName));
    x = x(:) - mean(x);

    idx1 = round(t_start*fs)+1;
    idx2 = round(t_end*fs);
    x = x(idx1:idx2);

    p = x * K_mic(mic);
    p_all{mic} = p;

    if doZ
        LAeq_Z(mic) = 10*log10(mean(p.^2)/p_ref^2);
    end

    if doA
        pA = Afilter(p);
        pA_all{mic} = pA;
        LAeq_A(mic) = 10*log10(mean(pA.^2)/p_ref^2);
    end

    if doC
        pC = Cfilter(p);
        pC_all{mic} = pC;
        LAeq_C(mic) = 10*log10(mean(pC.^2)/p_ref^2);
    end
end

%% ================== SPATIAL ENERGY AVERAGE ===================
energyMean = @(L) 10*log10(mean(10.^(L/10)));

LAeq_A_mean = energyMean(LAeq_A);
LAeq_C_mean = energyMean(LAeq_C);
LAeq_Z_mean = energyMean(LAeq_Z);

%% ================= ISO SUMMARY TABLE =========================
fprintf('\nISO 3382-2 BACKGROUND NOISE SUMMARY\n');
fprintf('--------------------------------------\n');
baseFileName = regexprep(fileName,'_100\d+\.wav$','');
fprintf('Measurement File:\n%s\n', baseFileName);
fprintf('--------------------------------------\n');

T_full = table((1:N)',LAeq_A',LAeq_C',LAeq_Z', ...
    'VariableNames',{'Mic','LAeq_dBA','LCeq_dBC','LZeq_dB'});

disp(T_full)

fprintf('\nSpatial Average:\n');
fprintf('LAeq = %.2f dB(A)\n',LAeq_A_mean);
fprintf('LCeq = %.2f dB(C)\n',LAeq_C_mean);
fprintf('LZeq = %.2f dB\n',LAeq_Z_mean);

%% ================= BUILD SPATIAL AVG SIGNALS =================
pZ_mat = cat(2,p_all{:});
pZ_avg = mean(pZ_mat,2);

if doA
    pA_mat = cat(2,pA_all{:});
    pA_avg = mean(pA_mat,2);
end
if doC
    pC_mat = cat(2,pC_all{:});
    pC_avg = mean(pC_mat,2);
end

%% ===================== FFT PLOT ===============================
if plotFFTAverage
    figure;
    hold on; grid on;

    L = length(pZ_avg);
    w = hann(L);
    Wcorr = sum(w)/L;

    [f,PZ] = localFFT(pZ_avg,fs,w,Wcorr,p_ref);
    semilogx(f,PZ);

    if doA
        [~,PA] = localFFT(pA_avg,fs,w,Wcorr,p_ref);
        semilogx(f,PA);
    end
    if doC
        [~,PC] = localFFT(pC_avg,fs,w,Wcorr,p_ref);
        semilogx(f,PC);
    end

    legend('Z','A','C');
    xlabel('Frequency [Hz]');
    ylabel('Magnitude [dB re 20 µPa]');
    xlim([10 fs/2]);
    ax = gca;
    ax.XAxis.Exponent = 0;  % disables the x10^N scaling
    ax.XScale = 'log';      % ensures x-axis is logarithmic
    ylim([-50 110]);
end
%% ================= PER-MIC FFT PLOTS =========================
if plotFFTAverage
    for mic = 1:N
        figure;
        hold on; grid on;

        L = length(p_all{mic});
        w = hann(L);
        Wcorr = sum(w)/L;

        % FFT of Z-weighted
        [f,PZ] = localFFT(p_all{mic},fs,w,Wcorr,p_ref);
        semilogx(f,PZ,'DisplayName','Z');

        % FFT of A-weighted
        if doA
            [~,PA] = localFFT(pA_all{mic},fs,w,Wcorr,p_ref);
            semilogx(f,PA,'DisplayName','A');
        end

        % FFT of C-weighted
        if doC
            [~,PC] = localFFT(pC_all{mic},fs,w,Wcorr,p_ref);
            semilogx(f,PC,'DisplayName','C');
        end

        xlabel('Frequency [Hz]');
        ylabel('Magnitude [dB re 20 µPa]');
        title(sprintf('Mic %d FFT',mic));
        legend show;
        xlim([10 fs/2]);
        ax = gca;
        ax.XAxis.Exponent = 0;  % disables the x10^N scaling
        ax.XScale = 'log';      % ensures x-axis is logarithmic
        ylim([-50 110]);
    end
end

%% ===================== OCTAVE BANDS ==========================
if doOctave
    Lmic_oct = zeros(N,length(fc_oct));

    for mic = 1:N
        y = FB_oct(pA_all{mic});
        Lmic_oct(mic,:) = 10*log10(mean(y.^2,1)/p_ref^2);
    end

    Lmean_oct = 10*log10(mean(10.^(Lmic_oct/10),1));

    T_oct = table(fc_oct',Lmean_oct', ...
        'VariableNames',{'CenterFrequency_Hz','LAeq_dBA'});

    fprintf('\nOCTAVE BAND LAeq (Spatial Average)\n');
    disp(T_oct)

    if plotOctaveBars
        figure;
        bar(fc_oct,Lmean_oct);
        set(gca,'XScale','log');
        xlabel('Hz'); ylabel('dB(A)');
        title('Octave Band LAeq');
        grid on;
    end
end

%% ================== 1/3 OCTAVE BANDS =========================
if doThirdOctave
    Lmic_third = zeros(N,length(fc_third));

    for mic = 1:N
        y = FB_third(pA_all{mic});
        Lmic_third(mic,:) = 10*log10(mean(y.^2,1)/p_ref^2);
    end

    Lmean_third = 10*log10(mean(10.^(Lmic_third/10),1));

    T_third = table(fc_third',Lmean_third', ...
        'VariableNames',{'CenterFrequency_Hz','LAeq_dBA'});

    fprintf('\n1/3 OCTAVE BAND LAeq (Spatial Average)\n');
    disp(T_third)

    if plotThirdOctaveBars
        figure;
        bar(fc_third,Lmean_third);
        set(gca,'XScale','log');
        xlabel('Hz'); ylabel('dB(A)');
        title('1/3 Octave Band LAeq');
        grid on;
    end
end

%% ===================== LOCAL FUNCTION ========================
function [f,SdB] = localFFT(sig,fs,w,Wcorr,p_ref)

L = length(sig);
X = fft(sig.*w);
X = X(1:floor(L/2)+1);
f = (0:floor(L/2))*fs/L;

mag = abs(X)/(L*Wcorr);
mag(2:end-1) = 2*mag(2:end-1);

SdB = 20*log10(mag/p_ref);
end
