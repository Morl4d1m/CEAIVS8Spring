%% ISO 3382-2 Processing (MLS + Sweep, Fully Functional)

load("C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\impulseResponseMeasurerData\omniSource\mlsAndSweepOmniSource25032026.mat")
load("C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\impulseResponseMeasurerData\omniPower\mlsAndSweep03032026.mat")

fs = 44100;

fc = [50 63 80 100 125 160 200 250 315 400 500 630 800 ...
      1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 ...
      10000 12500 16000 20000];
nBands = length(fc);

plotT20 = 1;
plotFFT = 1;
plotSchroeder = 1;
plotRT60box = 1;
savePlots = 1;
exportCSV = 0;
exportLatex = 0;
outputFolder = "C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\impulseResponseMeasurerData\omniSource";
plotFolder = fullfile(outputFolder, "impulseResponsePlots");
if ~exist(plotFolder, 'dir')
    mkdir(plotFolder);
end

%% Separate methods
%isSweep = strcmp(irdata_084724.Method,'Swept Sine');
%isMLS   = strcmp(irdata_084724.Method,'MLS');
isSweep = strcmp(irdata_120508.Method,'Swept Sine');
isMLS   = strcmp(irdata_120508.Method,'MLS');

methods  = {'Sweep','MLS'};
%datasets = {irdata_084724(isSweep,:), irdata_084724(isMLS,:)};
datasets = {irdata_120508(isSweep,:), irdata_120508(isMLS,:)};

for methodIdx = 1:2
    
    data = datasets{methodIdx};
    methodName = methods{methodIdx};
    
    fprintf('\n============================================\n');
    fprintf('Processing %s measurements\n', methodName);
    fprintf('============================================\n');
    
    nSpk = 8; nMic = 8;

    colors = lines(nSpk);
    speakerLabels = { ...
        'Speaker 1','Speaker 2','Speaker 3','Speaker 4', ...
        'Speaker 5','Speaker 6','Speaker 7','Speaker 8'};
    
    EDT  = zeros(nSpk,nMic,nBands);
    RT10 = zeros(nSpk,nMic,nBands);
    T20  = zeros(nSpk,nMic,nBands);
    T30  = zeros(nSpk,nMic,nBands);
    RT60 = zeros(nSpk,nMic,nBands);
    C50  = zeros(nSpk,nMic,nBands);
    C80  = zeros(nSpk,nMic,nBands);
    D50  = zeros(nSpk,nMic,nBands);
    D80  = zeros(nSpk,nMic,nBands);
    rows = {};
    rowCounter = 1;
    
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

        %% Store FFT data
        if plotFFT
            
            Nfft = 2^nextpow2(length(h));
            
            H = fft(h,Nfft);
            
            f = (0:Nfft/2-1)*fs/Nfft;
            
            FFTdata{spk,mic} = 20*log10(abs(H(1:Nfft/2)));
            
        end
        
        for b = 1:nBands
            
            of = octaveFilter(fc(b),'1/3 octave','SampleRate',fs);
            h_band = of(h);
            
            % Align to direct sound using 20% threshold
            [~, idx] = max(abs(h_band));
            h_band = h_band(idx:end);
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

            %% Store Schroeder curve (fullband example using 1 kHz band)
            if plotSchroeder && fc(b)==1000
                
                SchroederTime{spk,mic} = t;
                SchroederEDC{spk,mic} = edc_db;
                
            end

            % Remove noise floor
            N = length(edc_db);

            tailLength = min(1000, floor(N/3));   % use at most last third
            
            noiseFloor = mean(edc_db(N-tailLength+1:N));
            
            limit = noiseFloor + 10;   % 10 dB above noise
            
            validEnd = find(edc_db <= limit,1,'first');
            
            if ~isempty(validEnd) && validEnd > 10
                edc_db = edc_db(1:validEnd);
                t = t(1:validEnd);
            end
            
            % Decay parameters
            EDT(spk,mic,b)  = localRT(t,edc_db,0,-10);
            RT10(spk,mic,b) = localRT(t,edc_db,-5,-15);
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
            if exportCSV
    
                rows{rowCounter,1} = methodName;
                rows{rowCounter,2} = spk;
                rows{rowCounter,3} = mic;
                rows{rowCounter,4} = fc(b);
                
                rows{rowCounter,5} = EDT(spk,mic,b);
                rows{rowCounter,6} = RT10(spk,mic,b);
                rows{rowCounter,7} = T20(spk,mic,b);
                rows{rowCounter,8} = T30(spk,mic,b);
                rows{rowCounter,9} = RT60(spk,mic,b);
                
                rows{rowCounter,10} = C50(spk,mic,b);
                rows{rowCounter,11} = C80(spk,mic,b);
                rows{rowCounter,12} = D50(spk,mic,b);
                rows{rowCounter,13} = D80(spk,mic,b);
                
                rowCounter = rowCounter + 1;
                
            end
        if exportCSV
    
            resultsPerPair = cell2table(rows, ...
                'VariableNames', { ...
                'Method', ...
                'Speaker', ...
                'Microphone', ...
                'Frequency in Hz', ...
                'EDT in s', ...
                'RT10 in s', ...
                'RT20 in s', ...
                'RT30 in s', ...
                'RT60 in s', ...
                'C50 in dB', ...
                'C80 in dB', ...
                'D50', ...
                'D80'});
            
            filename = fullfile(outputFolder, ...
                sprintf('RoomAcoustics_%s_AllPairs.csv',methodName));
            
            writetable(resultsPerPair, filename);
            
            %fprintf('\nSaved per-measurement results to:\n%s\n', filename);
            
        end
        end
    end
    
    %% Spatial averages
    avgEDT  = squeeze(mean(mean(EDT ,'omitnan'),2,'omitnan'));
    avgRT10 = squeeze(mean(mean(RT10,'omitnan'),2,'omitnan'));
    avgT20  = squeeze(mean(mean(T20 ,'omitnan'),2,'omitnan'));
    avgT30  = squeeze(mean(mean(T30 ,'omitnan'),2,'omitnan'));
    avgRT60 = squeeze(mean(mean(RT60,'omitnan'),2,'omitnan'));
    avgC50  = squeeze(mean(mean(C50 ,'omitnan'),2,'omitnan'));
    avgC80  = squeeze(mean(mean(C80 ,'omitnan'),2,'omitnan'));
    avgD50  = squeeze(mean(mean(D50 ,'omitnan'),2,'omitnan'));
    avgD80  = squeeze(mean(mean(D80 ,'omitnan'),2,'omitnan'));
    
    stdT20  = std(reshape(T20,[],nBands),0,1);
    
    fc_col = fc(:);
    
    resultsTable = table( ...
        fc_col, avgEDT(:), avgRT10(:), avgT20(:), avgT30(:), avgRT60(:), ...
        avgC50(:), avgC80(:), avgD50(:), avgD80(:), ...% stdT20(:), ...
        'VariableNames',{'Frequency in Hz','EDT in S','RT10 in S','RT20 in S','RT30 in S','RT60 in S',...
        'C50 in dB','C80 in dB','D50','D80'});%,'Std_RT20 in S'});
    
    fprintf('\n---- ISO 3382-2 Spatially Averaged Results (%s) ----\n', methodName);
    disp(resultsTable)

    %% Export LaTeX table
    if exportLatex
    
        % Average across speakers (ISO practice)
        EDTmic  = squeeze(mean(EDT ,1,'omitnan'));   % [mic x band]
        T20mic  = squeeze(mean(T20 ,1,'omitnan'));
        RT60mic = squeeze(mean(RT60,1,'omitnan'));
        C50mic  = squeeze(mean(C50 ,1,'omitnan'));
        C80mic  = squeeze(mean(C80 ,1,'omitnan'));
        D50mic  = squeeze(mean(D50 ,1,'omitnan'));
    
        % File name
        latexFile = fullfile(outputFolder, sprintf('LatexTable_%s.txt',methodName));
    
        fid = fopen(latexFile,'w');
    
        if fid == -1
            error('Cannot create LaTeX file')
        end
    
        %% ---------- TABLE HEADER ----------
        fprintf(fid,'\\begin{sidewaystable}[p]\n');
        fprintf(fid,'\\centering\n');
        fprintf(fid,'\\renewcommand{\\arraystretch}{1.2}\n');
        fprintf(fid,'\\setlength{\\tabcolsep}{2.5pt}\n');
    
        fprintf(fid,'\\begin{tabular}{|p{1.6cm}|');
        for k = 1:22
            fprintf(fid,'p{0.9cm}|');
        end
        fprintf(fid,'}\n\\hline\n');
    
        fprintf(fid,'\\textbf{Speaker} & \\multicolumn{22}{c|}{\\textbf{OmniSource measurements averaged over all positions}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        %% ---------- FREQUENCY HEADER ----------
        freqList = [100 125 160 200 315 400 500 630 800 1000 1250 1600 ...
                    2000 3150 4000 5000 6300 8000 10000 12500 16000 20000];
    
        fprintf(fid,'&');
    
        for f = 1:length(freqList)
            fprintf(fid,' \\rotatebox{90}{%d Hz} &',freqList(f));
        end
    
        %fprintf(fid,' \\rotatebox{90}{White noise}');
        %fprintf(fid,' & \\rotatebox{90}{Pink Noise}');
        %fprintf(fid,' & \\rotatebox{90}{Sweep}');
        fprintf(fid,'\\\\\n');% & \\rotatebox{90}{16 bit MLS} \\\\\n');
    
        fprintf(fid,'\\hline\n');
    
        %% ---------- EDT ----------
        fprintf(fid,'\\textbf{Mic No.} & \\multicolumn{22}{c|}{\\textbf{Reverberation time (EDT) by signal type in seconds}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',EDTmic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
    
        end
    
        %% ---------- RT20 ----------
        fprintf(fid,'& \\multicolumn{22}{c|}{\\textbf{Reverberation time (RT20) by signal type in seconds}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',T20mic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
        end
    
        %% ---------- RT60 ----------
        fprintf(fid,'& \\multicolumn{22}{c|}{\\textbf{Reverberation time (RT60) by signal type in seconds}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',RT60mic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
    
        end
    
        fprintf(fid,'\\end{tabular}\n');
        fprintf(fid,'\\caption{Acoustic parameters averaged over all speaker positions.}\n');
        fprintf(fid,'\\label{tab:averageParametersOmniSource1%s}\n',methodName);
        fprintf(fid,'\\end{sidewaystable}\n');
    
        fprintf(fid,'\\begin{sidewaystable}[p]\n');
        fprintf(fid,'\\ContinuedFloat\n');
        fprintf(fid,'\\centering\n');
        fprintf(fid,'\\renewcommand{\\arraystretch}{1.2}\n');
        fprintf(fid,'\\setlength{\\tabcolsep}{2.5pt}\n');
    
        fprintf(fid,'\\begin{tabular}{|p{1.6cm}|');
        for k = 1:22
            fprintf(fid,'p{0.9cm}|');
        end
        fprintf(fid,'}\n\\hline\n');
    
        fprintf(fid,'\\textbf{Speaker} & \\multicolumn{22}{c|}{\\textbf{OmniSource measurements averaged over all positions}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        %% ---------- FREQUENCY HEADER ----------
        freqList = [100 125 160 200 315 400 500 630 800 1000 1250 1600 ...
                    2000 3150 4000 5000 6300 8000 10000 12500 16000 20000];
    
        %fprintf(fid,'&');
    
        for f = 1:length(freqList)
            fprintf(fid,' & \\rotatebox{90}{%d Hz}',freqList(f));
        end
    
        %fprintf(fid,' \\rotatebox{90}{White noise}');
        %fprintf(fid,' & \\rotatebox{90}{Pink Noise}');
        %fprintf(fid,' & \\rotatebox{90}{Sweep}');
        fprintf(fid,'\\\\\n');% & \\rotatebox{90}{16 bit MLS} \\\\\n');
    
        fprintf(fid,'\\hline\n');
    
        %% ---------- C50 ----------
        fprintf(fid,'\\textbf{Mic No.} & \\multicolumn{22}{c|}{\\textbf{Clarity (C50) by signal type in dB}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',C50mic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
    
        end
    
        %% ---------- C80 ----------
        fprintf(fid,'& \\multicolumn{22}{c|}{\\textbf{Clarity (C80) by signal type in dB}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',C80mic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
    
        end
    
        %% ---------- D50 ----------
        fprintf(fid,'& \\multicolumn{22}{c|}{\\textbf{Deutlichkeit/definition (D50) by signal type}} \\\\\n');
        fprintf(fid,'\\hline\n');
    
        for mic = 1:nMic
    
            fprintf(fid,'%d ',mic);
    
            for b = 1:length(freqList)
    
                idx = find(fc == freqList(b));
    
                if isempty(idx)
                    fprintf(fid,'& ');
                else
                    fprintf(fid,'& %.2f ',D50mic(mic,idx));
                end
    
            end
    
            fprintf(fid, ' \\\\\n');%'& & & & \\\\\n');
            fprintf(fid,'\\hline\n');
    
        end
    
        fprintf(fid,'\\end{tabular}\n');
        fprintf(fid,'\\caption{Acoustic parameters averaged over all speaker positions.}\n');
        fprintf(fid,'\\label{tab:averageParametersOmniSource2%s}\n',methodName);
        fprintf(fid,'\\end{sidewaystable}\n');
        
        fclose(fid);
    
        fprintf('LaTeX table saved to:\n%s\n',latexFile);
    
    end
    
    %% Plot
    if plotT20
        for spk = 1:nSpk
            fig = figure('Name',sprintf('%s - Speaker %d - T20',methodName,spk));
            hold on; grid on
            for mic = 1:nMic
                semilogx(fc, squeeze(T20(spk,mic,:)), ...
                'Color',colors(mic,:), ...
                'LineWidth',1.2)
            end
            set(gca,'XScale','log')   % force logarithmic axis
            
            % Explicit tick locations in Hz
            xticks([fc])
            xlabel('Frequency (Hz)')
            ylabel('T20 (s)')
            title(sprintf('%s - Speaker %d',methodName,spk))
        
            legend({'Mic1','Mic2','Mic3','Mic4','Mic5','Mic6','Mic7','Mic8'}, ...
                'Location','northeast')
            
            xlim([50 20000])     % dummy values
            ylim([0 0.4])          % dummy values

            if savePlots
                filename = fullfile(plotFolder,sprintf('%sT20Speaker%d.jpg', methodName, spk));
                exportgraphics(fig, filename, 'Resolution', 150);
            end
        end
    end

    %% FFT plots per microphone
    if plotFFT        
        for mic = 1:nMic
            figure('Name',sprintf('%s FFT Mic %d',methodName,mic))
            hold on
            grid on
            
            for spk = 1:nSpk                
                if isempty(FFTdata{spk,mic})
                    continue
                end                
                plot(f, FFTdata{spk,mic}, ...
                    'Color',colors(spk,:), ...
                    'LineWidth',1.2)                
            end
            
            set(gca,'XScale','log')   % force logarithmic axis
            
            % Explicit tick locations in Hz
            xticks([fc])
            
            % Force tick labels to appear as numbers instead of 10^x
            xtickformat('%.0f')
            xlabel('Frequency (Hz)')
            ylabel('Magnitude (dB)')
            title(sprintf('%s FFT - Microphone %d',methodName,mic))
            legend(speakerLabels,'Location','northeast')
            xlim([20 20000])     % dummy values
            ylim([-60 80])      % dummy values

            if savePlots
                filename = fullfile(plotFolder,sprintf('%sFFTMic%d.jpg', methodName, mic));
                exportgraphics(gcf, filename, 'Resolution', 150);
            end
        end
        
    end

    %% Schroeder decay plots
    if plotSchroeder
        
        for mic = 1:nMic
            
            figure('Name',sprintf('%s Schroeder Mic %d',methodName,mic))
            hold on
            grid on
            
            for spk = 1:nSpk
                
                if isempty(SchroederTime{spk,mic})
                    continue
                end
                
                plot(SchroederTime{spk,mic},SchroederEDC{spk,mic}, ...
                'Color',colors(spk,:), ...
                'LineWidth',1.2)
                
            end
            
            xlabel('Time (s)')
            ylabel('Level (dB)')
            title(sprintf('%s Schroeder Decay - Microphone %d',methodName,mic))
            
            title(sprintf('%s Schroeder Decay - Microphone %d',methodName,mic))
            
            legend(speakerLabels,'Location','northeast')
            yticks([-65 -60 -55 -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5])
            
            xlim([0 0.75])       % dummy values
            ylim([-65 5])     % dummy values

            if savePlots
                filename = fullfile(plotFolder,sprintf('%sSchroederMic%d.jpg', methodName, mic));
                exportgraphics(gcf, filename, 'Resolution', 150);
            end
        end
        
    end
    %% RT60 box plot averaged across speakers
    if plotRT60box
        
        % Average across frequency bands only
        RT60avgFreq = squeeze(mean(RT60,3,'omitnan'));   % size = [8 speakers x 8 mics]
        
        % Transpose so each column becomes a microphone group
        RT60box = RT60avgFreq';   % size = [8 speakers x 8 microphones]
        
        figure('Name',sprintf('%s RT60 Box Plot',methodName))
        
        boxplot(RT60box,'Labels',{'Mic1','Mic2','Mic3','Mic4','Mic5','Mic6','Mic7','Mic8'})
        
        xlabel('Microphone Number')
        ylabel('RT60 (s)')
        title(sprintf('%s Average RT60 per Microphone',methodName))
        
        grid on
        
        %xlim([0.5 8.5])
        %ylim([0 1.25])

        if savePlots
            filename = fullfile(plotFolder,sprintf('%sRT60Boxplot.jpg', methodName));
            exportgraphics(gcf, filename, 'Resolution', 150);
        end
    end
end

%% Linear regression helper
function T = localRT(t, edc_db, dB1, dB2)

% Select regression region
idx = find(edc_db <= dB1 & edc_db >= dB2);

if numel(idx) < 20
    T = NaN;
    return
end

% Dynamic range check
dynamicRange = max(edc_db) - min(edc_db);
requiredRange = abs(dB2);

if dynamicRange < requiredRange + 5
    T = NaN;
    return
end

% Linear regression
p = polyfit(t(idx), edc_db(idx),1);

% Correlation check
yfit = polyval(p,t(idx));
R = corrcoef(edc_db(idx), yfit);
r = R(1,2);

if r < 0.98
    T = NaN;
    return
end

% Compute decay time
decayRange = abs(dB2 - dB1);
T = -decayRange / p(1);

end