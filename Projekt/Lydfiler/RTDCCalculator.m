
%% ===================== USER SETTINGS =====================

basePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';
calFile  = fullfile(basePath,'micCalibrationConstants_94dB_1kHz_20022026.mat');

N_mics     = 8;
N_speakers = 4;

blockLength  = 6;      % seconds
signalLength = 1;      % seconds

sweep_f1 = 50;
sweep_f2 = 20000;
MLS_order = 16;

octBands = [125 250 500 1000 2000 4000 8000];

%% ============== TOGGLES =================

includeSines = true;

plotMeanEDT  = true;
plotMeanRT20 = true;
plotMeanRT60 = true;

%% ================= LOAD CALIBRATION =====================

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

%% ================= STORAGE ===============================

Results = {};
row = 1;

%% ================= MAIN LOOP =============================

for sourceType = ["omnipower","omnisource"]

    for spk = 1:N_speakers
        for mic = 1:N_mics

            filename = sprintf('%sSpot%d16022026__100%d.wav',...
                               sourceType,spk,mic);
            fullFile = fullfile(basePath,filename);

            if ~isfile(fullFile)
                continue
            end

            [x,fs] = audioread(fullFile);
            if size(x,2)>1
                x = mean(x,2);
            end

            x = x * K_mic(mic);
            x = x - mean(x);

            samplesPerBlock = blockLength * fs;
            samplesSignal   = signalLength * fs;
            nBlocks = floor(length(x)/samplesPerBlock);

            sineFreqs = [50 63 80 100 125 160 200 250 315 400 500 630 800 ...
                         1000 1250 1600 2000 2500 3150 4000 5000 6300 ...
                         8000 10000 12500 16000 20000];

            for b = 1:nBlocks

                blockStart = (b-1)*samplesPerBlock + 1;
                signalPart = x(blockStart:blockStart+samplesSignal-1);
                blockPart  = x(blockStart:blockStart+samplesPerBlock-1);
                decayPart  = x(blockStart+samplesSignal:blockStart+samplesPerBlock-1);

                if b <= length(sineFreqs)
                    if ~includeSines
                        continue
                    end
                    signalType = "Sine";
                    IR = decayPart;
                    band = sineFreqs(b);

                elseif b == length(sineFreqs)+1
                    signalType = "WhiteNoise";
                    IR = decayPart;

                elseif b == length(sineFreqs)+2
                    signalType = "PinkNoise";
                    IR = decayPart;

                elseif b == length(sineFreqs)+3
                    signalType = "Sweep";
                    IR = deconvolveSweep(signalPart,blockPart,fs,sweep_f1,sweep_f2);

                elseif b == length(sineFreqs)+4
                    signalType = "MLS";
                    IR = deconvolveMLS(signalPart,blockPart,MLS_order);

                else
                    continue
                end

                IR = IR - mean(IR);
                [~,idx] = max(abs(IR));
                IR = IR(idx:end);

                if signalType == "WhiteNoise" || signalType == "PinkNoise" ...
                   || signalType == "Sweep" || signalType == "MLS"

                    for ob = 1:length(octBands)
                        ir_f = octaveFilterISO(IR,fs,octBands(ob));
                        [Results,row] = storeResults(Results,row,...
                            ir_f,fs,sourceType,spk,mic,...
                            signalType,octBands(ob));
                    end
                else
                    [Results,row] = storeResults(Results,row,...
                        IR,fs,sourceType,spk,mic,...
                        signalType,band);
                end
            end
        end
    end
end

%% ================= TABLE & SAVE ===========================

T = cell2table(Results,...
'VariableNames',{'Source','Speaker','Mic','SignalType','Band',...
'EDT','RT20','RT30','RT60','C50','C80','D50','D80'});

writetable(T,fullfile(basePath,'ISO3382_Results.csv'));

%% ================= MICROPHONE STATISTICS =================

groupVars = {'Source','Speaker','SignalType','Band'};

Stats = groupsummary(T,groupVars,...
    {'mean','median','min','max'},...
    {'EDT','RT20','RT30','RT60','C50','C80','D50','D80'});

writetable(Stats,fullfile(basePath,'ISO3382_Statistics.csv'));

disp(Stats)

%% ================= MEAN DECAY CURVES =================

disp('Computing average decay curves...')

% Time axis for averaging (3 seconds is usually enough)
tCommon = linspace(0,3,3000);   % 1 ms resolution
signalTypes = unique(T.SignalType);

for sig = 1:length(signalTypes)

    thisType = signalTypes{sig};

    if strcmp(thisType,'Sine')
        continue   % skip narrowband sines
    end

    idxType = strcmp(T.SignalType,thisType);

    if sum(idxType)==0
        continue
    end

    decayMatrix = [];

    for i = find(idxType)'

        % Re-load corresponding IR
        source = T.Source{i};
        spk    = T.Speaker(i);
        mic    = T.Mic(i);
        band   = T.Band(i);

        filename = sprintf('%sSpot%d16022026__100%d.wav',...
                           source,spk,mic);
        fullFile = fullfile(basePath,filename);

        if ~isfile(fullFile)
            continue
        end

        [x,fs] = audioread(fullFile);
        if size(x,2)>1
            x = mean(x,2);
        end
        x = x * K_mic(mic);
        x = x - mean(x);

        % crude IR extraction (reuse Sweep logic only)
        IR = x;
        IR = IR - mean(IR);
        [~,peak] = max(abs(IR));
        IR = IR(peak:end);

        % Schroeder
        E = flipud(cumsum(flipud(IR.^2)));
        E = E / max(E);
        E_dB = 10*log10(E+eps);
        t = (0:length(E_dB)-1)/fs;

        % Interpolate to common axis
        E_interp = interp1(t,E_dB,tCommon,'linear',NaN);

        decayMatrix = [decayMatrix; E_interp];
    end

    if isempty(decayMatrix)
        continue
    end

    meanDecay = nanmean(decayMatrix,1);

    %% ===== PLOT =====
    figure
    plot(tCommon,meanDecay,'k','LineWidth',2)
    grid on
    xlabel('Time (s)')
    ylabel('Level (dB)')
    title(['Average Decay Curve - ' thisType])
    ylim([-80 5])
    xlim([0 3])
end


%% ================= CLEAN ISO PLOTS =================

metrics = {'mean_EDT','mean_RT20','mean_RT60'};
metricLabels = {'EDT (s)','RT20 (s)','RT60 (s)'};

uniqueSources = unique(string(Stats.Source));
uniqueSignals = unique(string(Stats.SignalType));

for m = 1:length(metrics)

    if (m==1 && ~plotMeanEDT) || ...
       (m==2 && ~plotMeanRT20) || ...
       (m==3 && ~plotMeanRT60)
        continue
    end

    for sig = 1:length(uniqueSignals)

        % Only plot broadband methods
        if uniqueSignals(sig) == "Sine"
            continue
        end

        figure
        hold on
        grid on

        for s = 1:length(uniqueSources)

            idx = strcmp(string(Stats.Source),uniqueSources(s)) & ...
                  strcmp(string(Stats.SignalType),uniqueSignals(sig));

            if sum(idx)==0
                continue
            end

            bands = Stats.Band(idx);
            values = Stats.(metrics{m})(idx);

            % Sort by frequency
            [bands,order] = sort(bands);
            values = values(order);

            semilogx(bands,values,'o-','LineWidth',1.5)
        end

        xlabel('Frequency (Hz)')
        ylabel(metricLabels{m})
        title(sprintf('%s - %s',uniqueSignals(sig),metricLabels{m}))
        legend(uniqueSources,'Location','best')
        xlim([min(octBands) max(octBands)])
    end
end

%% =========================================================
%% ===================== FUNCTIONS =========================
%% =========================================================

function IR = deconvolveSweep(exc,resp,fs,f1,f2)

    t = (0:length(exc)-1)'/fs;
    T = t(end);
    K = T/log(f2/f1);
    invSweep = flipud(exc).*exp(t/K);
    IR = fftfilt(invSweep,resp);
end

function IR = deconvolveMLS(exc,resp,order)

    N = 2^order - 1;

    if length(exc)<N
        IR = resp;
        return
    end

    H = fft(resp,N)./(fft(exc,N)+eps);
    IR = real(ifft(H));
end

function y = octaveFilterISO(x,fs,fc)

    f1 = fc/sqrt(2);
    f2 = fc*sqrt(2);

    if f1<=1, f1=1; end
    if f2>=fs/2-1, f2=fs/2-1; end
    if f1>=f2
        y=zeros(size(x));
        return
    end

    [b,a]=butter(3,[f1 f2]/(fs/2),'bandpass');
    y=filtfilt(b,a,x);
end

function [Results,row]=storeResults(Results,row,...
                                    ir,fs,...
                                    sourceType,spk,mic,...
                                    signalType,band)

    if max(abs(ir))==0
        return
    end

    ir = ir(:);
    ir = ir / max(abs(ir));

    % ===== Schroeder Integration =====
    E = flipud(cumsum(flipud(ir.^2)));
    E = E / max(E);
    E_dB = 10*log10(E + eps);
    t = (0:length(E_dB)-1)'/fs;

    % ===== Compute RT =====
    [EDT,RT20,RT30,RT60,fitEDT,fit20,fit30] = computeRT(t,E_dB);

    % ===== Clarity =====
    [C50,C80,D50,D80] = clarity(ir,fs);

    % ===== STORE =====
    Results(row,:)={char(sourceType),spk,mic,...
                    char(signalType),band,...
                    EDT,RT20,RT30,RT60,...
                    C50,C80,D50,D80};

    % ===== OPTIONAL DECAY PLOT =====
    if band==1000 && strcmp(signalType,'Sweep') && spk==1 && mic==1
        figure
        plot(t,E_dB,'k','LineWidth',1.5)
        hold on
        plot(t,fitEDT,'r--','LineWidth',1.5)
        plot(t,fit20,'b--','LineWidth',1.5)
        plot(t,fit30,'g--','LineWidth',1.5)
        grid on
        xlabel('Time (s)')
        ylabel('Level (dB)')
        title(sprintf('%s - %s - %d Hz',sourceType,signalType,band))
        legend('EDC','EDT fit','T20 fit','T30 fit')
        ylim([-80 5])
    end

    row=row+1;
end

function [EDT,RT20,RT30,RT60,fitEDT,fit20,fit30] = computeRT(t,E_dB)

    [EDT,fitEDT] = regression(t,E_dB,0,-10);
    [RT20,fit20] = regression(t,E_dB,-5,-25);
    [RT30,fit30] = regression(t,E_dB,-5,-35);

    if ~isnan(RT30)
        RT60 = RT30;
    else
        [RT60,~] = regression(t,E_dB,-5,-45);
    end
end

function [RT,fitLine] = regression(t,E_dB,upper,lower)

    idx = find(E_dB<=upper & E_dB>=lower);

    if length(idx)<10
        RT = NaN;
        fitLine = nan(size(E_dB));
        return
    end

    p = polyfit(t(idx),E_dB(idx),1);
    RT = -60/p(1);

    fitLine = polyval(p,t);
end

function [C50,C80,D50,D80]=clarity(ir,fs)

    t50=round(0.050*fs);
    t80=round(0.080*fs);

    E_total=sum(ir.^2);
    E_50=sum(ir(1:min(t50,end)).^2);
    E_80=sum(ir(1:min(t80,end)).^2);

    C50=10*log10(E_50/(E_total-E_50+eps));
    C80=10*log10(E_80/(E_total-E_80+eps));

    D50=100*(E_50/E_total);
    D80=100*(E_80/E_total);
end