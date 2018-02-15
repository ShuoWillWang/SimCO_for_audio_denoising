% MainTest - denoise an image
% this is a test file demonstrating how to denoise an image, 
% using learned dictionaries. The methods implemented here are the same
% one as described in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).

clear; close all;clc;

rand('state',0);
randn('state',0);

sizea = 256;
first = 0;

% Set initial input value
Param.k=512; % number of atoms in the dictionary
Param.noise = 10; % noise level
Param.method = 'SimCO';
ImageName = 'arctic_a0020.wav'; %image source
[Origin, fs]=audioread(ImageName); 
Origin = Origin(first + 1:min(first + sizea*sizea, end));
Noise = Origin;

OriginalImage = im2col(Origin,[sizea, 1],'distinct'); %reshape(Origin,513,[]);
NoisedImage = im2col(Noise,[sizea, 1],'distinct'); %reshape(Noise,513,[]);
% [OriginalImage, ~, ~] = STFTOther(Origin, 1024, 0.25, fs);
% [NoisedImage, ~, ~] = STFTOther(Noise, 1024, 0.25, fs);
val = max(max(abs(OriginalImage)));
OriginalImage = (OriginalImage) / val * 128;
NoisedImage = (NoisedImage) / val * 128 + 10 * randn(size(NoisedImage));

figure
imshow((OriginalImage + 128)/255)
figure
imshow((NoisedImage + 128)/255)

% pause(10000)

%time_start = clock;
% Denoise the corrupted image using learned dicitionary from corrupted image 
[DenoisedImage, timecost] = denoiseImage(NoisedImage, Param);
%time_end = clock;
%timecost = etime(time_end,time_start);

DenoisedImage = DenoisedImage + 128;
OriginalImage = OriginalImage + 128;
NoisedImage = NoisedImage + 128;

NoisedPSNR = 20*log10(255/sqrt(mean((NoisedImage(:)-OriginalImage(:)).^2)));
DenoisedPSNR = 20*log10(255/sqrt(mean((DenoisedImage(:)-OriginalImage(:)).^2)));

DenoisedImage = (DenoisedImage - 128) / 128 * val;
NoisedImage = (NoisedImage - 128) / 128 * val;

[mm nn] = size(NoisedImage);

Denoised = col2im(DenoisedImage,[sizea, 1],[mm*nn, 1],'distinct');
Noised = col2im(NoisedImage,[sizea, 1],[mm*nn, 1],'distinct');

audiowrite('arctic_a0020_noise.wav',Noised,fs)
audiowrite('arctic_a0020_denoised.wav',Denoised,fs)

figure
plot(Origin)
figure
plot(Noised)
figure
plot(Denoised)

if strcmp(Param.method, 'KSVD')
    save RealData_KSVD DenoisedPSNR DenoisedImage timecost;
end

if strcmp(Param.method, 'SimCO')
    save RealData_SimCO DenoisedPSNR DenoisedImage timecost;
end

if strcmp(Param.method, 'PSimCO')
    save RealData_PSimCO DenoisedPSNR DenoisedImage timecost;
end

if strcmp(Param.method, 'MOD')
    save RealData_MOD DenoisedPSNR DenoisedImage timecost;
end



% Display the results
figure;
subplot(1,3,1); imshow(OriginalImage,[]); title('Original clean image');
subplot(1,3,2); imshow(NoisedImage,[]); title(strcat(['Noisy image, ',num2str(NoisedPSNR),'dB']));
subplot(1,3,3); imshow(DenoisedImage,[]); title(strcat(['Denoised Image by trained dictionary, ',num2str(DenoisedPSNR),'dB']));
