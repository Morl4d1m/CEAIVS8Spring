%% ISO 3382-2 Processing (MLS + Sweep, Fully Functional)

load("C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\impulseResponseMeasurerData\mlsAndSweep03032026.mat")

fs = 44100;

fc = [50 63 80 100 125 160 200 250 315 400 500 630 800 ...
      1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 ...
      10000 12500 16000 20000];
nBands = length(fc);

plotFlag = 0;   % <<< TURN OFF HERE IF NEEDED

%% Separate methods
isSweep = strcmp(irdata_084724.Method,'Swept Sine');
isMLS   = strcmp(irdata_084724.Method,'MLS');

methods  = {'Sweep','MLS'};
datasets = {irdata_084724(isSweep,:), irdata_084724(isMLS,:)};

for methodIdx = 1:2
    
    data = datasets{methodIdx};
    methodName = methods{methodIdx};
    
    fprintf('\n============================================\n');
    fprintf('Processing %s measurements\n', methodName);
    fprintf('============================================\n');
    
    nSpk = 8; nMic = 8;
    
    EDT  = zeros(nSpk,nMic,nBands);
    RT10 = zeros(nSpk,nMic,nBands);
    T20  = zeros(nSpk,nMic,nBands);
    T30  = zeros(nSpk,nMic,nBands);
    RT60 = zeros(nSpk,nMic,nBands);
    C50  = zeros(nSpk,nMic,nBands);
    C80  = zeros(nSpk,nMic,nBands);
    D50  = zeros(nSpk,nMic,nBands);
    D80  = zeros(nSpk,nMic,nBands);
    
    %% Loop measurements
    for m = 1:height(data)
        
        spk = ceil(m/8);
        mic = mod(m-1,8)+1;
        
        % ALWAYS use stored impulse response
        h = data.ImpulseResponse(m).Amplitude;
        h = h(:);
        
        % Sanity check
        if all(h==0) || isempty(h)
            warning('Measurement %d contains zero IR',m)
            continue
        end
        
        % Remove DC and normalize
        h = h - mean(h);
        h = h / max(abs(h));
        
        for b = 1:nBands
            
            of = octaveFilter(fc(b),'1/3 octave','SampleRate',fs);
            h_band = of(h);
            
            % Align to direct sound using 20% threshold
            thr = 0.2*max(abs(h_band));
            idx = find(abs(h_band)>=thr,1,'first');
            if isempty(idx)
                continue
            end
            h_band = h_band(idx:end);
            
            h2 = h_band.^2;
            
            % Schroeder integration
            edc = flipud(cumsum(flipud(h2)));
            edc = edc / max(edc);
            edc_db = 10*log10(edc);
            t = (0:length(edc_db)-1)'/fs;
            
            % Decay parameters
            EDT(spk,mic,b)  = localRT(t,edc_db,0,-10);
            RT10(spk,mic,b) = localRT(t,edc_db,0,-10);
            T20(spk,mic,b)  = localRT(t,edc_db,-5,-25);
            T30(spk,mic,b)  = localRT(t,edc_db,-5,-35);
            
            % RT60 derivation
            if ~isnan(T30(spk,mic,b))
                RT60(spk,mic,b) = T30(spk,mic,b)*2;
            elseif ~isnan(T20(spk,mic,b))
                RT60(spk,mic,b) = T20(spk,mic,b)*3;
            else
                RT60(spk,mic,b) = NaN;
            end
            
            % Early/Late energy
            n50 = round(0.050*fs);
            n80 = round(0.080*fs);
            
            E_total = sum(h2);
            E50 = sum(h2(1:min(n50,end)));
            E80 = sum(h2(1:min(n80,end)));
            
            if E_total > 0
                C50(spk,mic,b) = 10*log10(E50/(E_total-E50));
                C80(spk,mic,b) = 10*log10(E80/(E_total-E80));
                D50(spk,mic,b) = E50/E_total;
                D80(spk,mic,b) = E80/E_total;
            end
        end
    end
    
    %% Spatial averages
    avgEDT  = squeeze(mean(mean(EDT ,1),2));
    avgRT10 = squeeze(mean(mean(RT10,1),2));
    avgT20  = squeeze(mean(mean(T20 ,1),2));
    avgT30  = squeeze(mean(mean(T30 ,1),2));
    avgRT60 = squeeze(mean(mean(RT60,1),2));
    avgC50  = squeeze(mean(mean(C50 ,1),2));
    avgC80  = squeeze(mean(mean(C80 ,1),2));
    avgD50  = squeeze(mean(mean(D50 ,1),2));
    avgD80  = squeeze(mean(mean(D80 ,1),2));
    
    stdT20  = std(reshape(T20,[],nBands),0,1);
    
    fc_col = fc(:);
    
    resultsTable = table( ...
        fc_col, avgEDT(:), avgRT10(:), avgT20(:), avgT30(:), avgRT60(:), ...
        avgC50(:), avgC80(:), avgD50(:), avgD80(:), stdT20(:), ...
        'VariableNames',{'Frequency_Hz','EDT_s','RT10_s','RT20_s','RT30_s','RT60_s',...
        'C50_dB','C80_dB','D50','D80','Std_RT20_s'});
    
    fprintf('\n---- ISO 3382-2 Spatially Averaged Results (%s) ----\n', methodName);
    disp(resultsTable)
    
    %% Plot
    if plotFlag
        for spk = 1:nSpk
            figure('Name',sprintf('%s - Speaker %d - T20',methodName,spk));
            hold on; grid on
            for mic = 1:nMic
                semilogx(fc, squeeze(T20(spk,mic,:)),'LineWidth',1.2)
            end
            set(gca,'XTick',fc)
            xlabel('Frequency (Hz)')
            ylabel('T20 (s)')
            title(sprintf('%s - Speaker %d',methodName,spk))
        end
    end
end

%% Linear regression helper
function RT = localRT(t, edc_db, dB1, dB2)
idx = find(edc_db <= dB1 & edc_db >= dB2);
if numel(idx) < 10
    RT = NaN;
    return
end
p = polyfit(t(idx), edc_db(idx),1);
RT = -60/p(1);
end