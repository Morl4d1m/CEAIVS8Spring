%% ISO 3382-2 Background Noise Measurement
% Optimized + ISO compliant tables
% Author: Christian Lykke


%% =========================== FLAGS ===========================
doZ = true;
doA = true;
doC = true;
doOctave      = 0;
doThirdOctave = 0;
plotFFTAverage = 1;
plotFFTPerMic  = 1;  
plotOctaveBars = 0;
plotThirdOctaveBars = 0;
batchProcessMeasurements = 0;
doStatisticsAndExport = 0;
savePlots = 1;

%% =========================== PATHS ===========================
basePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
%calFile  = fullfile(basePath,'micCalibrationConstants_94dB_1kHz_20022026.mat');
plotPath = fullfile(basePath,'noisePlots');
if savePlots && ~exist(plotPath,'dir')
    mkdir(plotPath);
end

%% ================= BATCH FILE LIST =================
fileList = {
'DEAAbning02022026kl1415-1431'
'DEAAbning02022026kl1448-1459MangeIBaren'
'DEAAbning02022026kl1536-1547Tits'
'DEAAbning02022026kl1632-1640IndenMusik'
'DEFoersteFredag06022026Snestorm'
'DEFoersteFredag06022026Snestorm2'
'poolSpilMedLidtBaggrundsSnak13022026'
'poolOgBordFåMennesker13022026'
'DEAndenFredag13022026'
'DEAndenFredag130220262'
'DEAndenFredag13022026Tits'
'DEAndenFredag13022026Beerpong'
'DETredjeFredag20022026kl1306-1317'
'DETredjeFredag20022026kl1330-1344'
'DETredjeFredag20022026kl1410-1424'
'DETredjeFredag20022026kl1432-1452'
'DETredjeFredag20022026kl1519-1529'
'DETredjeFredag20022026JengaTits'
'DEFjerdeFredag27022026'
'DEFjerdeFredag27022026kl14-1420'
'DEFjerdeFredag27022026kl1427-1442'
'DEFemteFredag06032026kl1311-1323'
'DEFemteFredag06032026kl1337-1357'
'DEFemteFredag06032026kl1408-1415'
'DEFemteFredag06032026kl1424-1445'
'DESjetteFredag13032026kl1301-1317'
'DESjetteFredag13032026kl1322-1335'
'DESjetteFredag13032026kl1339-1356'
'DESyvendeFredag20032026kl1300-1312'
'DESyvendeFredag20032026kl1324-1339'
'DESyvendeFredag20032026kl1355-1401'
'DESyvendeFredag20032026kl1410-1417Tits'
'DESyvendeFredag20032026kl1420-1501Beerpong'
'DESyvendeFredag20032026kl1518-1535'
'DESyvendeFredag20032026kl1539-1547'
'DESyvendeFredag20032026kl1613-1625Vihygger'
'DESyvendeFredag20032026kl1717-1734'
'DESyvendeFredag20032026kl1744-1844'
'backgroundMeasurement13022026'
'backgroundMeasurement13032026'
'backgroundMeasurement16022026'
'backgroundMeasurement18022026'
'backgroundMeasurement20022026'
'backgroundMeasurement27022026'
'backgroundMeasurement28022026'
'backgroundMeasurement03032026'
'backgroundMeasurement06032026'
'backgroundMeasurement20032026'
};

lombardTest = {
'DELombard55dB27032026'
'DELombard60dB27032026'
'DELombard65dB27032026'
'DELombard70dB27032026'
'DELombard75dB27032026'
};

backgroundNoise = {    
'backgroundMeasurement13022026'
'backgroundMeasurement13032026'
'backgroundMeasurement16022026'
'backgroundMeasurement18022026'
'backgroundMeasurement20022026'
'backgroundMeasurement27022026'
'backgroundMeasurement28022026'
'backgroundMeasurement03032026'
'backgroundMeasurement06032026'
'backgroundMeasurement20032026'
};


if batchProcessMeasurements
    %measurementList=fileList;
    measurementList=lombardTest;
    %measurementList=backgroundNoise;
else
    measurementList= {'completeOpenNoise28032026'};
end
resultTable = table;

%% ================= CONSTANTS =================
N = 8;
p_ref = 20e-6;
fc = [20 25 31.5 40 50 63 80 100 125 160 200 250 315 400 500 630 800 ...
      1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 ...
      10000 12500 16000 20000];
fs=44100;

%% ==================== LOAD CALIBRATION =======================
%S = load(calFile);

%if isfield(S,'K_mic')
%    K_mic = S.K_mic;
%elseif isfield(S,'K')
%    K_mic = S.K;
%elseif isfield(S,'cal')
%    K_mic = [S.cal.mic.K_PaPerFS];
%else
%    error('Calibration constants not found.');
%end



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

%% ================= MEASUREMENT LOOP =================

for meas = 1:length(measurementList)

baseFileName = measurementList{meas};

fprintf('\n====================================\n');
fprintf('Processing measurement:\n%s\n',baseFileName);

%% ================= LOAD CALIBRATION =================

dateToken = regexp(baseFileName,'\d{8}','match','once');

if isempty(dateToken)
    error('Date not found in filename: %s',baseFileName)
end

calFile = fullfile(basePath, ...
    sprintf('micCalibrationConstants_94dB_1kHz_%s.mat',dateToken));

S = load(calFile);

if isfield(S,'K_mic')
    K_mic = S.K_mic;
elseif isfield(S,'K')
    K_mic = S.K;
elseif isfield(S,'cal')
    K_mic = [S.cal.mic.K_PaPerFS];
else
    error('Calibration constants not found.')
end

%% ====================== MAIN LOOP ============================
for mic = 1:N
    fileName = sprintf('%s__100%d.wav',baseFileName,mic);
    [x, ~] = audioread(fullfile(basePath,fileName));
    x = x(:) - mean(x);
    
    if contains(baseFileName, "background", 'IgnoreCase', true)
        t_start = 0;
    else
        t_start = 6*60*60+20*60;           % seconds
    end
    t_end   = 6*60*60+20*60+70*60; %length(x)/fs; % full file length

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
fprintf('LAeq = %.3f dB(A)\n',LAeq_A_mean);
fprintf('LCeq = %.3f dB(C)\n',LAeq_C_mean);
fprintf('LZeq = %.3f dB\n',LAeq_Z_mean);

resultTable = [resultTable; table( ...
string(baseFileName), ...
LAeq_A_mean, ...
LAeq_C_mean, ...
LAeq_Z_mean, ...
'VariableNames',{'Measurement','LAeq_dBA','LCeq_dBC','LZeq_dB'})];

%% ================= STORE MIC DATA =================

if meas == 1
    micTable = table;
end

for mic = 1:N
    micTable = [micTable; table( ...
        string(baseFileName), ...
        mic, ...
        LAeq_A(mic), ...
        LAeq_C(mic), ...
        LAeq_Z(mic), ...
        'VariableNames',{'Measurement','Mic','LAeq_dBA','LCeq_dBC','LZeq_dB'})];
end

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
set(0,'DefaultFigureVisible','off')
%% ===================== FFT PLOT ===============================
if plotFFTAverage
    figure;
    hold on; grid on;

    L = length(pZ_avg);
    w = hann(L);
    Wcorr = sum(w)/L;

    [f,PZ] = localFFT(pZ_avg,fs,w,Wcorr,p_ref);
    semilogx(f,PZ);
    if doC
        [~,PC] = localFFT(pC_avg,fs,w,Wcorr,p_ref);
        semilogx(f,PC);
    end

    if doA
        [~,PA] = localFFT(pA_avg,fs,w,Wcorr,p_ref);
        semilogx(f,PA);
    end
    plotName=regexprep(baseFileName, '_', '');
    title(sprintf('Average FFT of measurement: %s', plotName), 'Interpreter','none');
    legend('Z','C','A');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    xticks([fc])
    xtickformat('%.0f');
    xlim([20 fs/2]);
    ax = gca;
    ax.XAxis.Exponent = 0;  % disables the x10^N scaling
    ax.XScale = 'log';      % ensures x-axis is logarithmic
    ylim([-70 70]);
    savePlot(savePlots, plotPath, sprintf('FFTAverage%s.jpg',plotName));
end


%% ================= PER-MIC FFT PLOTS =========================
if plotFFTPerMic
    for mic = 1:N
        figure;
        hold on; 
        grid on;

        L = length(p_all{mic});
        w = hann(L);
        Wcorr = sum(w)/L;

        % Z-weighted FFT
        [f,PZ] = localFFT(p_all{mic},fs,w,Wcorr,p_ref);
        semilogx(f,PZ,'DisplayName','Z');

        % C-weighted FFT
        if doC
            [~,PC] = localFFT(pC_all{mic},fs,w,Wcorr,p_ref);
            semilogx(f,PC,'DisplayName','C');
        end

        % A-weighted FFT
        if doA
            [~,PA] = localFFT(pA_all{mic},fs,w,Wcorr,p_ref);
            semilogx(f,PA,'DisplayName','A');
        end

        xlabel('Frequency (Hz)');
        ylabel('Magnitude (dB)');
        title(sprintf('Microphone %d FFT Spectrum',mic));

        legend show;

        xlim([20 fs/2]);
        xticks([fc])
        xtickformat('%.0f');

        ax = gca;
        ax.XAxis.Exponent = 0;
        ax.XScale = 'log';

        ylim([-70 70]);
        plotName=regexprep(baseFileName, '_', '');
        savePlot(savePlots, plotPath, sprintf('FFTMic%d%s.jpg',mic,plotName));
    end
end
set(0,'DefaultFigureVisible','on')
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
        plotName=regexprep(baseFileName, '_', '');
        savePlot(savePlots, plotPath, sprintf('Octaves%s.jpg',plotName));
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
        plotName=regexprep(baseFileName, '_', '');
        savePlot(savePlots, plotPath, sprintf('thirdOctaves%s.jpg',plotName));
    end
end

%% ================= FINAL BATCH SUMMARY =================

if batchProcessMeasurements
    fprintf('\n====================================\n');
    fprintf('BATCH MEASUREMENT SUMMARY\n');
    disp(resultTable)
end

%% ================= STATISTICS + CSV EXPORT =================

if doStatisticsAndExport && batchProcessMeasurements

    fprintf('\n====================================\n');
    fprintf('EXPORTING DATA + COMPUTING STATISTICS\n');

    %% -------- Identify background measurements --------
    isBackground = contains(resultTable.Measurement,"background","IgnoreCase",true);

    resultTable.Group = repmat("NonBackground",height(resultTable),1);
    resultTable.Group(isBackground) = "Background";

    micTable.Group = repmat("NonBackground",height(micTable),1);
    micTable.Group(contains(micTable.Measurement,"background","IgnoreCase",true)) = "Background";

    %% -------- Save raw data --------
    writetable(resultTable, fullfile(basePath,'SpatialAverages_AllMeasurements.csv'));
    writetable(micTable, fullfile(basePath,'PerMic_AllMeasurements.csv'));

    %% -------- Helper function --------
    statFun = @(x)[mean(x), std(x), min(x), max(x)];

    %% ================= SPATIAL STATS =================
    statsSpatial = table;

    groups = ["Background","NonBackground"];

    for g = 1:length(groups)
        idx = resultTable.Group == groups(g);

        if any(idx)
            LA = resultTable.LAeq_dBA(idx);

            s = statFun(LA);

            statsSpatial = [statsSpatial; table( ...
                groups(g), ...
                s(1), s(2), s(3), s(4), ...
                'VariableNames',{'Group','Mean_dBA','Std_dB','Min_dB','Max_dB'})];
        end
    end

    %% ================= MIC STATS =================
    statsMic = table;

    for g = 1:length(groups)
        for mic = 1:N

            idx = micTable.Group == groups(g) & micTable.Mic == mic;

            if any(idx)
                LA = micTable.LAeq_dBA(idx);

                s = statFun(LA);

                statsMic = [statsMic; table( ...
                    groups(g), mic, ...
                    s(1), s(2), s(3), s(4), ...
                    'VariableNames',{'Group','Mic','Mean_dBA','Std_dB','Min_dB','Max_dB'})];
            end
        end
    end

    %% -------- Save statistics --------
    writetable(statsSpatial, fullfile(basePath,'Stats_Spatial.csv'));
    writetable(statsMic, fullfile(basePath,'Stats_PerMic.csv'));

    %% -------- Display --------
    fprintf('\nSpatial Statistics:\n');
    disp(statsSpatial)

    fprintf('\nPer-Microphone Statistics:\n');
    disp(statsMic)

end

function savePlot(savePlots, plotPath, fileName)
if savePlots
    exportgraphics(gcf, fullfile(plotPath, fileName), 'Resolution', 150);
    close(gcf);
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