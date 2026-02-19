%% MATLAB Audio Player - Teensy Equivalent (FLAG CONTROL + PINK NOISE)
% Plays (individually selectable):
% 1) 1/3 octave sine tones (50 Hz to 20 kHz) with per-tone gain
% 2) White noise
% 3) Pink noise
% 4) Log sine sweep
% 5) MLS (LFSR-based, identical taps)
%
% Output: laptop soundcard

clear; clc; close all;

%% ==========================
% USER SETTINGS
%% ==========================
fs = 44100;              % Sample rate (Hz)
masterGain = 0.11;       % Similar to mixer.gain() in Teensy

toneDuration = 1.0;      % seconds
silenceAfterTone = 5.0;  % seconds

noiseDuration = 1.0;     % seconds
silenceAfterNoise = 5.0; % seconds

sweepStart = 50;         % Hz
sweepEnd = 20000;        % Hz
sineSweepTime = 0.743;   % seconds
silenceAfterSweep = 5.0; % seconds

silenceBeforeNextBlock = 0.5;

LFSRBits = 16;           % same as Teensy
samplesPerBit = 1;       % same as playMLSBit default
mlsAmplitude = 1;      % relative to full scale

%% ==========================
% FLAGS (TURN SIGNALS ON/OFF)
%% ==========================
PLAY_SINE_TONES  = true;
PLAY_WHITE_NOISE = true;
PLAY_PINK_NOISE  = true;
PLAY_SINE_SWEEP  = true;
PLAY_MLS         = true;

%% ==========================
% PER-SINE GAIN CONTROL
% One gain value for each frequency index f=7..33
%% ==========================
toneGain = ones(1, 33);   % default flat
toneGain = max(0, min(toneGain, 1));

%% ==========================
% AUDIO OUTPUT OBJECT
%% ==========================
deviceWriter = audioDeviceWriter( ...
    'SampleRate', fs, ...
    'SupportVariableSizeInput', true);

%% ==========================
% MAIN SEQUENCE
%% ==========================
count = 1;
fprintf("Iteration #%d\n", count);
count = count + 1;

%% ==========================
% 1) 1/3 OCTAVE SINE TONES
%% ==========================
if PLAY_SINE_TONES
    for f = 7:33
        freq = octaveFrequency(f);

        fprintf("Sine at %d Hz\n", freq);

        t = (0:1/fs:toneDuration-1/fs).';
        x = sin(2*pi*freq*t);

        x = x .* (toneGain(f) * masterGain);

        xStereo = [x x];
        playBlocking(deviceWriter, xStereo);

        playSilence(deviceWriter, fs, silenceAfterTone);
    end

    playSilence(deviceWriter, fs, silenceBeforeNextBlock);
end

%% ==========================
% 2) WHITE NOISE
%% ==========================
if PLAY_WHITE_NOISE
    fprintf("White noise\n");

    N = round(noiseDuration * fs);
    noise = randn(N,1);
    noise = noise / max(abs(noise));
    noise = noise * masterGain;

    noiseStereo = [noise noise];
    playBlocking(deviceWriter, noiseStereo);

    playSilence(deviceWriter, fs, silenceAfterNoise);
end

%% ==========================
% 3) PINK NOISE
%% ==========================
if PLAY_PINK_NOISE
    fprintf("Pink noise\n");

    N = round(noiseDuration * fs);
    pink = generatePinkNoiseVoss(N);

    pink = pink / max(abs(pink));
    pink = pink * masterGain;

    pinkStereo = [pink pink];
    playBlocking(deviceWriter, pinkStereo);

    playSilence(deviceWriter, fs, silenceAfterNoise);
end

%% ==========================
% 4) SINE SWEEP (LOG)
%% ==========================
if PLAY_SINE_SWEEP
    fprintf("Sine sweep\n");

    t = (0:1/fs:sineSweepTime-1/fs).';
    sweep = chirp(t, sweepStart, sineSweepTime, sweepEnd, 'logarithmic');
    sweep = sweep * masterGain;

    sweepStereo = [sweep sweep];
    playBlocking(deviceWriter, sweepStereo);

    playSilence(deviceWriter, fs, silenceAfterSweep);
end

%% ==========================
% 5) MLS
%% ==========================
if PLAY_MLS
    generateMLS_and_play(deviceWriter, fs, LFSRBits, samplesPerBit, mlsAmplitude, masterGain);
end

fprintf("Done\n");
pause(5);
fprintf("Disconnect\n");
pause(5);

release(deviceWriter);

%% ============================================================
% FUNCTIONS
%% ============================================================

function playBlocking(deviceWriter, xStereo)
    frameSize = 1024;
    idx = 1;
    N = size(xStereo,1);

    while idx <= N
        i2 = min(idx + frameSize - 1, N);
        deviceWriter(xStereo(idx:i2,:));
        idx = i2 + 1;
    end
end

function playSilence(deviceWriter, fs, durationSec)
    if durationSec <= 0
        return;
    end
    N = round(durationSec * fs);
    silence = zeros(N,2);
    playBlocking(deviceWriter, silence);
end

function freq = octaveFrequency(octaveBand)
    switch octaveBand
        case 1, freq = 13;
        case 2, freq = 16;
        case 3, freq = 20;
        case 4, freq = 25;
        case 5, freq = 32;
        case 6, freq = 40;
        case 7, freq = 50;
        case 8, freq = 63;
        case 9, freq = 80;
        case 10, freq = 100;
        case 11, freq = 125;
        case 12, freq = 160;
        case 13, freq = 200;
        case 14, freq = 250;
        case 15, freq = 315;
        case 16, freq = 400;
        case 17, freq = 500;
        case 18, freq = 630;
        case 19, freq = 800;
        case 20, freq = 1000;
        case 21, freq = 1250;
        case 22, freq = 1600;
        case 23, freq = 2000;
        case 24, freq = 2500;
        case 25, freq = 3150;
        case 26, freq = 4000;
        case 27, freq = 5000;
        case 28, freq = 6300;
        case 29, freq = 8000;
        case 30, freq = 10000;
        case 31, freq = 12500;
        case 32, freq = 16000;
        case 33, freq = 20000;
        otherwise
            error("Unsupported octave band!");
    end
end

function taps = feedbackTaps(bits)
    switch bits
        case 2, taps = bitor(bitshift(1,1), bitshift(1,0));
        case 3, taps = bitor(bitshift(1,2), bitshift(1,0));
        case 4, taps = bitor(bitshift(1,3), bitshift(1,0));
        case 5, taps = bitor(bitshift(1,4), bitshift(1,2));
        case 6, taps = bitor(bitshift(1,5), bitshift(1,4));
        case 7, taps = bitor(bitshift(1,6), bitshift(1,5));
        case 8, taps = bitor(bitor(bitshift(1,7), bitshift(1,5)), bitor(bitshift(1,4), bitshift(1,3)));
        case 9, taps = bitor(bitshift(1,8), bitshift(1,4));
        case 10, taps = bitor(bitshift(1,9), bitshift(1,6));
        case 11, taps = bitor(bitshift(1,10), bitshift(1,8));
        case 12, taps = bitor(bitor(bitshift(1,11), bitshift(1,5)), bitor(bitshift(1,3), bitshift(1,0)));
        case 13, taps = bitor(bitor(bitshift(1,12), bitshift(1,3)), bitor(bitshift(1,2), bitshift(1,0)));
        case 14, taps = bitor(bitor(bitshift(1,13), bitshift(1,12)), bitor(bitshift(1,11), bitshift(1,1)));
        case 15, taps = bitor(bitshift(1,14), bitshift(1,13));
        case 16, taps = bitor(bitor(bitshift(1,15), bitshift(1,13)), bitor(bitshift(1,12), bitshift(1,10)));
        case 17, taps = bitor(bitshift(1,16), bitshift(1,13));
        case 18, taps = bitor(bitshift(1,17), bitshift(1,10));
        case 19, taps = bitor(bitor(bitshift(1,18), bitshift(1,17)), bitor(bitshift(1,16), bitshift(1,13)));
        case 20, taps = bitor(bitshift(1,19), bitshift(1,16));
        case 21, taps = bitor(bitshift(1,20), bitshift(1,18));
        case 22, taps = bitor(bitshift(1,21), bitshift(1,20));
        case 23, taps = bitor(bitshift(1,22), bitshift(1,17));
        case 24, taps = bitor(bitor(bitshift(1,23), bitshift(1,22)), bitor(bitshift(1,21), bitshift(1,16)));
        case 25, taps = bitor(bitshift(1,24), bitshift(1,21));
        case 26, taps = bitor(bitor(bitshift(1,25), bitshift(1,5)), bitor(bitshift(1,1), bitshift(1,0)));
        case 27, taps = bitor(bitor(bitshift(1,26), bitshift(1,4)), bitor(bitshift(1,1), bitshift(1,0)));
        case 28, taps = bitor(bitshift(1,27), bitshift(1,24));
        case 29, taps = bitor(bitshift(1,28), bitshift(1,26));
        case 30, taps = bitor(bitor(bitshift(1,29), bitshift(1,5)), bitor(bitshift(1,3), bitshift(1,0)));
        case 31, taps = bitor(bitshift(1,30), bitshift(1,27));
        case 32, taps = bitor(bitor(bitshift(1,31), bitshift(1,21)), bitor(bitshift(1,1), bitshift(1,0)));
        otherwise
            error("Unsupported bit length!");
    end
end

function generateMLS_and_play(deviceWriter, fs, LFSRBits, samplesPerBit, mlsAmplitude, masterGain)

    if LFSRBits < 2 || LFSRBits > 32
        error("LFSRBits must be between 2 and 32.");
    end

    mask = uint32(bitshift(1, LFSRBits) - 1);
    LFSR = mask;
    taps = uint32(feedbackTaps(LFSRBits));

    MLSLength = double(bitshift(1, LFSRBits) - 1);

    fprintf("Generating MLS with %d bits:\n", LFSRBits);
    fprintf("The MLS should be %d bits long.\n", MLSLength);

    tic;

    bits = false(MLSLength, 1);

    for i = 1:MLSLength
        x = bitand(LFSR, taps);
        feedback = mod(sum(bitget(x, 1:32)), 2) == 1;

        bits(i) = feedback;

        LFSR = bitshift(LFSR, 1);
        if feedback
            LFSR = bitor(LFSR, uint32(1));
        end
        LFSR = bitand(LFSR, mask);
    end

    elapsed = toc;
    fprintf("MLS generation complete.\n");
    fprintf("It has taken %.6f seconds to calculate.\n", elapsed);

    mls = double(bits);
    mls(mls == 0) = -1;
    mls(mls == 1) = +1;

    if samplesPerBit > 1
        mls = repelem(mls, samplesPerBit);
    end

    mls = mls * mlsAmplitude * masterGain;

    mlsStereo = [mls(:) mls(:)];

    fprintf("Playing MLS...\n");
    playBlocking(deviceWriter, mlsStereo);
end

function pink = generatePinkNoiseVoss(N)
    % Simple Voss-McCartney pink noise generator (no toolbox required)
    % Produces approximately 1/f noise.

    numRows = 16;  % more rows = better pinkness
    array = randn(numRows, 1);
    runningSum = sum(array);

    pink = zeros(N, 1);
    counter = zeros(numRows, 1);

    for i = 1:N
        % Determine which rows to update (binary carry)
        n = i;
        row = 1;

        while bitand(n, 1) == 0
            n = bitshift(n, -1);
            row = row + 1;
            if row > numRows
                break;
            end
        end

        if row <= numRows
            runningSum = runningSum - array(row);
            array(row) = randn();
            runningSum = runningSum + array(row);
        end

        pink(i) = runningSum;
    end
end
