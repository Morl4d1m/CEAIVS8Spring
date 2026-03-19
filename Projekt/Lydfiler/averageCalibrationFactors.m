%% Path
basePath = 'C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\';

%% Find all calibration MAT files
files = dir(fullfile(basePath,'micCalibrationConstants_94dB_1kHz_*.mat'));

if isempty(files)
    error('No calibration files found.');
end

N = 8; % number of microphones
nFiles = length(files);

K_all = zeros(nFiles,N);
dates = strings(nFiles,1);

%% Load all calibration files
for i = 1:nFiles
    
    filePath = fullfile(basePath, files(i).name);
    
    data = load(filePath,'K_mic');
    
    if ~isfield(data,'K_mic')
        error('File %s does not contain K_mic', files(i).name);
    end
    
    K_all(i,:) = data.K_mic(:)';
    
    % Extract date from filename
    name = files(i).name;
    d = regexp(name,'\d{8}','match'); % finds DDMMYYYY
    if ~isempty(d)
        dates(i) = d{1};
    end
    
end

%% Compute averages
K_mean = mean(K_all,1);
K_std  = std(K_all,0,1);

%% Display results
fprintf('\nAverage calibration constants (Pa/FS)\n');
fprintf('-------------------------------------\n');

for mic = 1:N
    fprintf('Mic %d: mean = %.6f   std = %.6f\n', mic, K_mean(mic), K_std(mic));
end

K_mic = mean(K_all,1);

save('micCalibrationConstants_average.mat','K_mic')