% "Solutions" for Day 1 "Spectral analysis of spiking
% responses of sensory neurons" of the G-Node Winter course in Neural Data
% Analysis, 2017
% This script uses cells that can be run separately. Uses Xcorr, 
% which is part of the signal processing toolbox!
% by Jan Grewe, Uni Tuebingen

%% Looking at the raw data
clear
close all

% the p-unit data
load('data_p-unit.mat');
samplerate      = 20000; % Hz
duration        = 10; %seconds
time            =  1/samplerate:1/samplerate:duration;  % time vector

% preparations for a rasterplot
[spikesX,spikesY] = find(responses_strong);    % find spike indices
spikesX = spikesX./samplerate;             % convert x indices to times

% create a psth
kernel = gaussKernel(0.001, 1./samplerate)'; % Gauss kernel with 1 ms std
cResponse = [];
for(i=1:size(responses_strong,2))
    cResponse(:,i) = conv(responses_strong(:,i),kernel);
end

psth = mean(cResponse,2);
psth(1:(ceil(length(kernel)/2)-1))      =[];
psth(end-(floor(length(kernel)/2)-1):end)   =[];

% create cross correlogram
[c,lags] = xcorr(stimulus_strong(:,2),psth,2000,'coeff');


figure
subplot(4,1,1)
plot(time,stimulus_strong);
title ('Amplitude modulation stimulus');
xlabel('time [s]');
ylabel('contrast');
xlim([0 .1]);

subplot(4,1,2)
plot(spikesX,spikesY,'.','MarkerSize',2);
title('rasterplot')
xlabel('time [s]');
ylabel('trial no');
ylim([0 size(responses_strong,2)+1])
xlim([0 0.1]);

subplot(4,1,3)
plot(time, psth)
title('psth');
xlabel('time [s]');
ylabel('firing frequency [Hz]');
xlim([0 0.1]);

subplot(4,1,4)
plot(lags./samplerate,c);
title('cross correlogram between stimulus and response');
xlabel('lag [s]')
ylabel('correlation coefficient');

%% Optimal kernel width of a gaussian kernel or a boxkernel
% Which kernel width or bandwidth of a box kernel is the right one? 
% Apply the methods by Shimazaki and Shinomoto, 2007 and/or 
% Shimazaki and Shinomoto, 2010

% 1: convert the binary spikes to spike times
time = stimulus_strong(:, 1); % a time axis
spike_times = []; 
for i = 1:size(responses_strong,2)
times = time(responses_strong(:,i) == 1);
spike_times = cat(1, spike_times, times);
end

% 2: simply apply the methods
% a) use the sskernel method for Gaussian kernels
[estimate, ~, optw] = sskernel(spike_times, time);

figure
plot(time, estimate);
xlabel('time [s]');
ylabel('firing rate [Hz]')
disp(strcat('optimal kernel width: ', num2str(optw)));

% b) user the optimal bandwidth method for box kernels
[bandwidths, costs] = optimal_kernel_bandwidth(spike_times, size(responses_strong, 2));
optimal_bandwidth = bandwidths(costs == min(costs));

disp(strcat('optimal boxwidth kernel width: ', num2str(optimal_bandwidth)));
plot(bandwidths*1000, costs)
xlabel('bin width [ms]')
ylabel('C_n')

% apply the optimal bandwidth kernel and plot the estimate... 

%% Power Spectrum I - playground
clear
% close all

% A) effects of data segmentation

% create some noisy data
samplerate      = 4096;                     % Hz
duration        = 10;                       % seconds
time            = 1/samplerate:1/samplerate:duration;  % time vector
frequencies     = [0.5 1 2 5 7 10 12 25 50 100 150 200 202 250];
amplitudes      = ones(size(frequencies)).*0.75;
segmentLengths  = [512 1024 4096 8192 32768];

y = zeros(size(time));                      % combine sinusoids
for (i=1:length(frequencies))
    y = y + amplitudes(i)*sin(2*pi*frequencies(i)*time);
end
y = y + 2*randn(size(time));                % add some noise

% plot the trace
figure
plot(time,y)
title('artificial data')
xlabel('time [s]')
ylabel('intensity [arb. units]');

% estimate power spectra with different segment lengths

for (i=1:length(segmentLengths))  % cycle through segment lengths
    f               = [0 (samplerate/segmentLengths(i):samplerate/segmentLengths(i):samplerate/2)];
    noOfSamples		= length(y);
    noOfSegments	= floor(noOfSamples/segmentLengths(i));
    powers          = zeros(length(f),noOfSegments);    % some space for the resulting power spectra.
    
    for (j=1:noOfSegments) % create psd for each segment
        start	= (j-1)*segmentLengths(i)+1;
        ende	= start+segmentLengths(i)-1;
        segment	= y(start:ende);
        py      = abs(fft(segment,segmentLengths(i))).^2; % the power spectrum
        powers(:,j) = py(1:floor(length(py)/2)+1);        % single-sided spectrum (positive freqs only)
    end
    
    figure
    plot(f,mean(powers,2));
    title(sprintf('average power spectrum using segments of %d samples length. Duration of dataset: %d seconds @ %d Hz'...
        ,segmentLengths(i),duration,samplerate));
    xlabel('frequency [Hz]');
    ylabel('power');
end

%% Power spectrum II - real data
% close all

% p-unit data
clear
load('data_p-unit.mat');
samplerate      = 20000; % Hz
duration        = 10; %seconds
segmentLength   = 4096; %samples
time            = 1/samplerate:1/samplerate:duration;  % time vector
f               = [0 (samplerate/segmentLength:samplerate/segmentLength:samplerate/2)]; %frequency vector
noOfSamples		= size(responses_strong,1);
noOfSegments	= floor(noOfSamples/segmentLength);
window          = hann(segmentLength);

%stimulus power
powers          = zeros(length(f),noOfSegments);    % some space for the resulting power spectra
for (j=1:noOfSegments) % create psd for each segment
    start	= (j-1)*segmentLength+1;
    ende	= start+segmentLength-1;
    segment	= stimulus_strong(start:ende,2);
    segment = segment - mean(segment,1);
    pSegment= abs(fft(segment.*window,segmentLength)).^2; % the power spectrum
    powers(:,j) = pSegment(1:floor(length(pSegment)/2)+1);  % single-sided spectrum (positive freqs only)
end
stimulusPower = mean(powers,2);

%response power
powers          = zeros(length(f),noOfSegments);    % some space for the resulting power spectra
for (j=1:noOfSegments) % create psd for each segment
    start	= (j-1)*segmentLength+1;
    ende	= start+segmentLength-1;
    segment	= responses_strong(start:ende,2); %only of the first response
    segment = segment - mean(segment,1);
    pSegment= abs(fft(segment.*window,segmentLength)).^2; % the power spectrum
    powers(:,j) = pSegment(1:floor(length(pSegment)/2)+1);  % single-sided spectrum (positive freqs only)
end
responsePower = mean(powers,2);

figure
% plot results
subplot(2,1,1)
loglog(f,stimulusPower);
xlim([1 1000])
ylim([100 100000])
title('stimulus power spectrum')
xlabel('frequency [Hz]')
ylabel('power')

subplot(2,1,2)
plot(f,responsePower);
xlim([1 1000])
ylim([1 1000])
title ('response power spectrum -- strange high freq power due to dirac like spikes...');
xlabel('frequency [Hz]')
ylabel('power')


% convolve responses with kernels of different size
kernelWidths = [0.25 1 2 5 10]./1000; %kernel stds in ms
convolvedResponsePower = zeros(length(f),length(kernelWidths));
for(i=1:length(kernelWidths))
    kernel = gaussKernel(kernelWidths(i), 1/samplerate)';
    cresp = [];
    for(j=1:size(responses_strong,2))
        cresp(:,j) = conv(responses_strong(:,j),kernel);
    end
    psth = mean(cresp,2);
    
    powers          = zeros(length(f),noOfSegments);    % some space for the resulting power spectra
    for (j=1:noOfSegments) % create psd for each segment
        start	= (j-1)*segmentLength+1;
        ende	= start+segmentLength-1;
        segment	= psth(start:ende);
        segment = segment - mean(segment,1);
        pSegment= abs(fft(segment.*window,segmentLength)).^2; % the power spectrum
        powers(:,j) = pSegment(1:floor(length(pSegment)/2)+1);  % single-sided spectrum (positive freqs only)
    end
    convolvedResponsePower(:,i) = mean(powers,2);
end

figure
semilogy(repmat(f',1,size(convolvedResponsePower,2)),convolvedResponsePower);
title('response power spectra using different kernels')
xlabel('frequency [Hz]');
ylabel('power');
xlim([1 1000])
ylim([0.0001 1000])

%% Coherence
clear
% close all

load('data_p-unit.mat')
samplerate      = 20000; % Hz
segmentLength   = 16768; %samples
f               = [0 (samplerate/segmentLength:samplerate/segmentLength:samplerate/2)]; %frequency vector
noOfSamples		= size(responses_strong,1);
noOfSegments	= floor(noOfSamples/segmentLength);
stimulus        = stimulus_weak(:,2)-mean(stimulus_weak(:,2),1); %remove the mean stimulus intensity
kernel          = gaussKernel(0.00025, 1/samplerate)';
cresp           = []; % temporal variable for the convolved response
window          = hann(segmentLength);
averageResponse = conv(mean(responses_strong,2), kernel,'same');

c = zeros(segmentLength,size(responses_strong,2));
for(i=1:size(responses_strong,2)) 
    cresp           = conv(responses_strong(:,i),kernel);
    fResp           = zeros(segmentLength,noOfSegments);
    fStim           = zeros(segmentLength,noOfSegments);
    for (j=1:noOfSegments) % create fft for each segment
        start           = (j-1)*segmentLength+1;
        ende            = start+segmentLength-1;
        rSegment        = cresp(start:ende);%response segment
        stimSegment     = stimulus(start:ende); %stimulus segment
        fResp(:,j)      = fft(rSegment.*window,segmentLength); % the Fourier spectrum of the response segement
        fStim(:,j)      = fft(stimSegment.*window,segmentLength);% the Fourier spectrum of the stimulus segment
    end
    
    fRespConj = conj(fResp); %complex conjugate of the average response spectrum
    fStimConj = conj(fStim); %complex conjugate of the average stimulus spectra
    
    srCrossSpectrum = mean(fStimConj .* fResp,2); %cross spectrum S*R
    ssAutoSpectrum  = mean(fStimConj .* fStim,2); %auto spectrum S*S
    
    rsCrossSpectrum = mean(fRespConj .* fStim,2); %cross spectrum R*S
    rrAutoSpectrum  = mean(fRespConj .* fResp,2); %auto spectrum R*R
    c(:,i) = (srCrossSpectrum .* rsCrossSpectrum) ./ (ssAutoSpectrum .* rrAutoSpectrum);
end

ec = zeros(segmentLength,size(responses_strong,2));
for(i=1:size(responses_strong,2)) 
    cresp           = conv(responses_strong(:,i),kernel);
    fResp           = zeros(segmentLength,noOfSegments);
    fStim           = zeros(segmentLength,noOfSegments);
    for (j=1:noOfSegments) % create fft for each segment
        start           = (j-1)*segmentLength+1;
        ende            = start+segmentLength-1;
        rSegment        = cresp(start:ende);%response segment
        stimSegment     = averageResponse(start:ende); %stimulus segment
        fResp(:,j)      = fft(rSegment,segmentLength); % the Fourier spectrum of the response segement
        fStim(:,j)      = fft(stimSegment,segmentLength);% the Fourier spectrum of the stimulus segment
    end
    
    fRespConj = conj(fResp); %complex conjugate of the average response spectrum
    fStimConj = conj(fStim); %complex conjugate of the average stimulus spectra
    
    srCrossSpectrum = mean(fStimConj .* fResp,2); %cross spectrum S*R
    ssAutoSpectrum  = mean(fStimConj .* fStim,2); %auto spectrum S*S
    
    rsCrossSpectrum = mean(fRespConj .* fStim,2); %cross spectrum R*S
    rrAutoSpectrum  = mean(fRespConj .* fResp,2); %auto spectrum R*R
    ec(:,i) = (srCrossSpectrum .* rsCrossSpectrum) ./ (ssAutoSpectrum .* rrAutoSpectrum);
end


figure
plot(f,mean(c(1:length(f),:),2),'r','DisplayName','stimulus-response coherence');
hold on;
plot(f,mean(ec(1:length(f),:),2),'k','DisplayName','response-response coherence');
xlim([0 1000])
ylim([0 1])
legend('show')
title ('single-sided coherence between stimulus and p-unit responses')
xlabel('frequency [Hz]')
ylabel('\gamma^2')


%*************************************************************************%
% expected coherence is calculated between the individual responses and   %
% the average response which is considered "noise free". Thus, the ex-    %
% pected coherence can be used to separate effects due to noise and non   %
% linearity of the system.                                                %
%*************************************************************************%

%% Reverse reconstruction using the spike triggered average (optional)
clear
load('data_p-unit.mat')
samplerate      = 20000; % Hz
stimulus        = stimulus_strong(:,2)-mean(stimulus_strong(:,2),1); %remove the mean stimulus intensity
time            = (1:length(stimulus))/samplerate;  % time vector
interval        = 40*(samplerate/1000); % let's consider some ms before and after the spike
sta             = zeros(2*interval+1,1); %some space to store the STA
sta_time        = (-interval:interval)./samplerate;
count           = 0;
tic
for(i = 1:size(responses_strong,2))
    spike_times = find(responses_strong(:,i)==1); % in samples, not seconds
    for(j = 1:length(spike_times)) % for all spikes
        index = spike_times(j);
        if(index+interval <= length(stimulus) && index - interval >= 1) % consider only those spikes which fit into the stimulus vector
            sta   = sta + stimulus(index-interval:index+interval);
            count = count + 1;
        end
    end
end
sta = sta./count;


temp = []; % temporary variable to store individual estimates
for(i = 1: size(responses_strong,2))
    temp(:,i) = conv(responses_strong(:,i),sta,'same');
end

s_est = mean(temp,2);
% s_est(1:interval) = []; % cut away the convolution rubbish
% s_est(end-interval+1:end) = [];

t_elapsed = toc;

figure
plot(sta_time.*1000,sta);
title ('spike-triggered average')
xlabel('time [ms]')
ylabel('stimulus')

figure
plot(time,s_est,'r','LineWidth',1.25,'DisplayName','estimated stimulus')
hold on
plot(time,stimulus,'k','LineWidth',1,'DisplayName','real stimulus')
title (strcat('comparison of stimulus and reconstructed stimulus, elapsed time: ',num2str(t_elapsed),'s'))
legend('show')
xlim([1.0 1.25])
ylim([-1 1])
xlabel('time [s]')
ylabel('stimulus')


%% Reverse reconstruction using reverse filters (optional)
clear
% close all

load('data_p-unit.mat')
samplerate      = 20000; % Hz
segmentLength   = 4096; %samples
duration        = 10; %s
f               = [0 (samplerate/segmentLength:samplerate/segmentLength:samplerate/2)]; %frequency vector
time            = 1/samplerate:1/samplerate:duration;  % time vector
noOfSamples		= size(responses_strong,1);
noOfSegments	= floor(noOfSamples/segmentLength);
stimulus        = stimulus_strong(:,2)-mean(stimulus_strong(:,2),1); %remove the mean stimulus intensity
kernel          = gaussKernel(0.00025, 1/samplerate)';
cresp           = []; % temporal variable for the convolved response
win             = hann(segmentLength*2);
pad             = zeros(segmentLength/2,1);
tic

temp = zeros(segmentLength * 2, size(responses_strong, 2)); %temporary variable for individual reverse filter 
for(i=1:size(responses_strong, 2))
    cresp           = conv(responses_strong(:,i), kernel);
    fResp           = zeros(segmentLength * 2, noOfSegments);
    fStim           = zeros(segmentLength * 2, noOfSegments);
    for (j=1:noOfSegments) % create fft for each segment
        start           = (j-1) * segmentLength + 1;
        ende            = start + segmentLength - 1;
        rSegment        = cat(1, pad, cresp(start:ende), pad);%response segment
        stimSegment     = cat(1, pad, stimulus(start:ende), pad); %stimulus segment
        fResp(:,j)      = fft(rSegment .* win, segmentLength * 2); % the Fourier spectrum of the response segement
        fStim(:,j)      = fft(stimSegment .* win, segmentLength * 2);% the Fourier spectrum of the stimulus segment
    end
    fRespConj = conj(fResp); %complex conjugate of the average response spectrum
    rsCrossSpectrum = mean(fRespConj .* fStim, 2); %cross spectrum R*S
    rrAutoSpectrum  = mean(fRespConj .* fResp, 2); %auto spectrum R*R
    temp(:,i)  = rsCrossSpectrum ./ rrAutoSpectrum;
end
h_rev = mean(temp, 2); % the reverse filter H(f)

%apply filter to response to reconstruct the stimulus from the responses
responseSpectrum = fft(mean(responses_strong(1:segmentLength * 2, :), 2));

s_est = ifft(h_rev .* responseSpectrum);

t_elapsed = toc;

% plot the estimated stimulus normalized to the maximum, maybe apply some
% smoothing
figure
plot(time(1:segmentLength*2), s_est/max(s_est), 'r');
hold on
plot(time(1:segmentLength*2), stimulus(1:segmentLength*2), 'k')
title (strcat('comparison of stimulus and reconstructed stimulus, elapsed time: ', num2str(t_elapsed),'s'))
xlabel('time [s]')
ylabel('intensity ')


%*************************************************************************%
% forward filter can be calculated analogously H_fwd(f) = <S*R>/<S*S>     %
%                                                                         %
% from the spike data one could do the reconstruction using the STA       %
%*************************************************************************%

