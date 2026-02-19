%% Solutions Topic 1
clear all; close all; clc;  %Clear memory, close figures and clean control window.

%% Problem 1
% Using the signalAnalyzer app it is possible to find the frequency components in the spectrum to approx. 
% 311 Hz, 392 Hz and 466 Hz. But we can't see when they appear and for how long. The pitch of these three
% frequencies are associated with the musical tones, D#, G and A#.

%% Problem 2
% In order to find out when and for how long the different tones appear we must use a spectrogram
% Now we need to use the spectrogram function in order to get a good resolution as compared to the one in
% signalAnalyzer

%Lets first import the audio
[audiodata_exercise1,Fs]=audioread('Exercise1.wav');

% Parameters for the spectrogram
windowlength=4096; %time window length
window=hann(windowlength,"periodic"); %Creating a Hanning window It must be periodic for proper spectrograms and overlap add.
Blocksize=windowlength;  %Choose same FFT blocksize as the window length
Overlapsamples=windowlength/2;   %Choose 50% overlap
used_colormap='parula';  %Colormap for the spectrogram

% As i find some plotting issues with the default figure from spetrogram() when changing to logarithmic frequency axis. 
% Therefore,  I extract the calculated STFT and do my own figure.
[freq_content,freq_vector,time_vector]=spectrogram(audiodata_exercise1,window,Overlapsamples,Blocksize,Fs,'yaxis','MinThreshold',-70);

%Create a new figure for my own plot
figure;                     %create a new figure
freq_content_dB=20*log10(abs(freq_content));                                    %Here we make the frequency content logarithmic on a dB scale
surfhandle=surf(time_vector,freq_vector,freq_content_dB,freq_content_dB(:,:));  %Here i call the surf function which creates a 3D surface
axis xy;                    %Set time to x axis, frequency to y-axis
colormap(used_colormap);    %Use the specified color axis
shading interp              %Interpolate colors between points in spectrogram - this really helps giving the surface a "nice" look
clim([-30 50]);             %Color limits - can help reduce the visual influence of noise etc.
set(gca,'YScale','log');    %Make y-axis logarithmic
xlabel('Time [s]');         %Label for x axis
ylabel('Frequency [Hz]');   %Label for y axis
zlabel('Magnitude [dB]');   %Label for z-axis (color axis)
ylim([20 20000]);           %Specify data range for the frequency axis
set(gca,'View',[0 90]);     %See plot from the top - it can still be rotated, but seen from the top we see it as a 2D picture.
bar_handle=colorbar;        %Plot a colorbar next to the spectrogram
bar_handle.Label.String='Magnitude [dB]';   %Make a label for the colorbar

%% Problem3
% For problem 3 i reuse much of the code from problem 2
% You could put the spectrogram code into a function in Matlab in order to make your script much shorter and
% nicer. But I will let you do that yourself. Here i just use the script to generate the spectrogram

[audiodata_exercise3,Fs]=audioread('Exercise3.wav');
% Parameters for the spectrogram
windowlength=4096; %time window length
window=hann(windowlength,"periodic"); %Creating a Hanning window It must be periodic for proper spectrograms and overlap add.
Blocksize=windowlength;  %Choose same FFT blocksize as the window length
Overlapsamples=windowlength/2;   %Choose 50% overlap
used_colormap='parula';  %Colormap for the spectrogram

% As i find some plotting issues with the default figure from spetrogram() when changing to logarithmic frequency axis. 
% Therefore,  I extract the calculated STFT and do my own figure.
[freq_content_ex3,freq_vector_ex3,time_vector_ex3]=spectrogram(audiodata_exercise3,window,Overlapsamples,Blocksize,Fs,'yaxis','MinThreshold',-70);

%Create a new figure for my own plot
figure;                     %create a new figure
freq_content_dB_ex3=20*log10(abs(freq_content_ex3));                                    %Here we make the frequency content logarithmic on a dB scale
surfhandle_ex3=surf(time_vector_ex3,freq_vector_ex3,freq_content_dB_ex3,freq_content_dB_ex3(:,:));  %Here i call the surf function which creates a 3D surface
axis xy;                    %Set time to x axis, frequency to y-axis
colormap(used_colormap);    %Use the specified color axis
shading interp              %Interpolate colors between points in spectrogram - this really helps giving the surface a "nice" look
clim([10 35]);              %Color limits - can help reduce the visual influence of noise etc.
set(gca,'YScale','log');    %Make y-axis logarithmic
xlabel('Time [s]');         %Label for x axis
ylabel('Frequency [Hz]');   %Label for y axis
zlabel('Magnitude [dB]');   %Label for z-axis (color axis)
ylim([20 20000]);           %Specify data range for the frequency axis
set(gca,'View',[0 90]);     %See plot from the top - it can still be rotated, but seen from the top we see it as a 2D picture.
bar_handle=colorbar;        %Plot a colorbar next to the spectrogram
bar_handle.Label.String='Magnitude [dB]';   %Make a label for the colorbar

% If you inspect the spectrogram you will find a similar as exercise 2 pattern starting close to 10 seconds
% into the audio. But notice lots of harmonics and if you really look close you may see that it is actually a
% a tone lower - from a to g on the musical scale. 

%% Problem 4
% The solution to this problem depends on your choice of signals, but changing the windowlength parameter in
% the above used code to 256, 1024 and 4096 makes different choices on the tradeoff between time/frequency "uncertainty" in the resulting
% spectrograms. 