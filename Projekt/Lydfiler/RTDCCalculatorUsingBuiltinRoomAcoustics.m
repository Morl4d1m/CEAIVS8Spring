%% Room Acoustics Analysis with Struct Impulse Responses

% --------------------------- User Settings -----------------------------
matFilePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\impulseResponseMeasurerData\mlsAndSweep03032026.mat';
plotFlag = true; % Set to true to generate heatmaps
numSpeakers = 8;
numMics = 8;
% -----------------------------------------------------------------------

% Load the .mat file
dataStruct = load(matFilePath);
irTable = dataStruct.irdata_084724; 

% Identify unique methods (e.g., 'MLS', 'Sweep')
methods = unique(irTable.Method);

for m = 1:length(methods)
    methodName = methods{m};
    fprintf('======================= Method: %s =======================\n', methodName);
    
    % Filter table by method
    methodTable = irTable(strcmp(irTable.Method, methodName), :);
    
    % Preallocate results table
    results = table();
    results.Speaker = zeros(height(methodTable),1);
    results.Microphone = zeros(height(methodTable),1);
    results.RT60 = zeros(height(methodTable),1);
    results.EDT = zeros(height(methodTable),1);
    results.C50 = zeros(height(methodTable),1);
    results.D50 = zeros(height(methodTable),1);
    results.TS = zeros(height(methodTable),1); % Strength

    % Loop over measurements
    for idx = 1:height(methodTable)

    speakerPos = ceil(idx / numMics);
    micPos     = mod(idx-1, numMics) + 1;

    % --- Extract IR ---
    irStruct = methodTable.ImpulseResponse(idx);
    ir = irStruct.Amplitude;
    fs = methodTable.SampleRate(idx);

    % --- Convert to double ---
    ir = double(ir(:));

    % --- Remove DC ---
    ir = ir - mean(ir);

    % --- Normalize ---
    if max(abs(ir)) > 0
        ir = ir ./ max(abs(ir));
    end

    % --- Find direct sound peak ---
    [~, peakIdx] = max(abs(ir));

    % Keep 1 ms before peak
    preSamples = round(0.001 * fs);
    startIdx = max(1, peakIdx - preSamples);

    ir = ir(startIdx:end);

    % --- Remove trailing noise (optional but helps stability) ---
    energy = cumsum(ir.^2,'reverse');
    energy = energy ./ max(energy);
    cutoffIdx = find(energy < 1e-6, 1);

    if ~isempty(cutoffIdx)
        ir = ir(1:cutoffIdx);
    end

    % --- Skip if too short ---
    if length(ir) < fs * 0.2
        warning('IR %d too short after trimming. Skipping.', idx);
        continue
    end

    % --- Room acoustics ---
    try
        raParams = roomacoustics(ir, fs, ...
            'FrequencyBands','octave', ...
            'T30',true);
    catch ME
        warning('roomacoustics failed for measurement %d: %s', idx, ME.message);
        continue
    end

    results.Speaker(idx) = speakerPos;
    results.Microphone(idx) = micPos;

    % If octave band output, take 1 kHz band (typical comparison band)
    if isstruct(raParams.RT60)
        [~, bandIdx] = min(abs(raParams.CenterFrequencies - 1000));
        results.RT60(idx) = raParams.RT60.T30(bandIdx);
        results.EDT(idx)  = raParams.EDT(bandIdx);
        results.C50(idx)  = raParams.C50(bandIdx);
        results.D50(idx)  = raParams.D50(bandIdx);
        results.TS(idx)   = raParams.Strength(bandIdx);
    else
        results.RT60(idx) = raParams.RT60;
        results.EDT(idx)  = raParams.EDT;
        results.C50(idx)  = raParams.C50;
        results.D50(idx)  = raParams.D50;
        results.TS(idx)   = raParams.Strength;
    end

end

    % Display results in command window
    disp(results);

    % Optional plotting
    if plotFlag
        figure('Name',['Room Acoustics Heatmaps - ' methodName],'NumberTitle','off');
        
        % RT60 Heatmap
        subplot(1,3,1);
        rt60Mat = reshape(results.RT60, numMics, numSpeakers)'; % speakers as rows
        imagesc(rt60Mat); colorbar; axis square;
        title('RT60 (s)'); xlabel('Mic Position'); ylabel('Speaker Position');
        set(gca,'YDir','normal');

        % EDT Heatmap
        subplot(1,3,2);
        edtMat = reshape(results.EDT, numMics, numSpeakers)'; 
        imagesc(edtMat); colorbar; axis square;
        title('EDT (s)'); xlabel('Mic Position'); ylabel('Speaker Position');
        set(gca,'YDir','normal');

        % C50 Heatmap
        subplot(1,3,3);
        c50Mat = reshape(results.C50, numMics, numSpeakers)'; 
        imagesc(c50Mat); colorbar; axis square;
        title('C50 (dB)'); xlabel('Mic Position'); ylabel('Speaker Position');
        set(gca,'YDir','normal');
    end
end