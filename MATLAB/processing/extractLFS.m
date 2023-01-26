function [ieegLFS, ieegLFSP] = extractBeta(data, fs, fDown, tw, gtw, name, normFactor, normType)
    % Extracts low-frequency signal (LFS) of ieeg (active) with normalization factors
    % if provided. LFS is the input signal (assumed to be raw ieeg) filtered with
    % an anti-aliasing filter and downsampled to the specified frequency.
    %
    % Input:
    % data - ieeg data channels x trials x timepoints
    % fDown - Downsampled frequency (Optional; if not present use
    %   same sampling frequency)
    % tw - time window of the data
    % gtw - output time-window
    % name - label for data (auditory, response)
    % normFactor - normalization values (channels x 2; if not
    %   present, no normalization
    % normType - Normalization type (1=zscore, 2=mean subtracted)
    % 
    % Output:
    % ieegLFS - Extracted spike band structure;
    % ieegLFSP - Power of extracted spike band

switch nargin
    case 6
        normFactor = [];
        normType = 1;
    case 7
        normType = 1;
end

disp(['Extracting LFS ' name])
ieegLFSTemp = [];

for iTrial = 1:size(data, 2)
    if (fs~=fDown)
        dh2 = resample(double(squeeze(data(:, iTrial, :)))', fDown, fs)';
    end
    timeDown = linspace(tw(1),tw(2),size(dh2,2));
    eTime = timeDown>=gtw(1)&timeDown<=gtw(2);
    % Normalization
    if(~isempty(normFactor))
        if(normType==1)
        dh2 = (dh2(:,eTime)-normFactor(:,1))./normFactor(:,2); 
        end
        if(normType==2)
            dh2 = dh2(:,eTime)-normFactor(:,1);
        end
    else
        dh2 = dh2(:,eTime);
    end
    ieegLFSTemp(:, iTrial, :) = dh2;
end


ieegLFS = ieegStructClass(ieegLFSTemp, fDown, gtw, [0, fDown/2], name);
ieegLFSP = (squeeze(mean(log10(ieegLFSTemp.^2), 3)));

end