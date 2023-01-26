function [ieegSpikeBand, ieegSBP] = extractSBP_chestek(data, fs, fDown, tw, gtw, name, normFactor, normType)
    % Extracts spike band power (SBP) of ieeg (active) with normalization factors
    % if provided
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
    % ieegSpikeBand - Extracted spike band structure;
    % ieegSPB - Power of extracted spike band

switch nargin
    case 6
        normFactor = [];
        normType = 1;
    case 7
        normType = 1;
end

disp(['Extracting Spike Band (Chestek) ' name])
fSB = [300 1000]; % Hz
nFilterPoints = 200; % Filter order
% decimateFactor = 10; % Decimate to fs/10 (20 kHz -> 2 kHz usually)
decimateFactor = round(fs/fDown)
ieegSBtemp = [];

if size(data, 1) == 1
    ieegSBtemp(1, :, :) = ...
        EcogExtractSpikeBandPowerTrial(double(squeeze(data)), fs, fSB, tw,...
                                       gtw, nFilterPoints, decimateFactor,...
                                       normFactor, normType);
        % EcogExtractHighGammaTrial(double(squeeze(data)), fs, fDown, fSB, tw,...
        %                           gtw, normFactor, normType);

else
    for iTrial = 1:size(data, 2)
        ieegSBtemp(:, iTrial, :) = ...
        EcogExtractSpikeBandPowerTrial(double(squeeze(data(:, iTrial,:))),...
                                       fs, fSB, tw, gtw, nFilterPoints,...
                                       decimateFactor, normFactor, normType);
            % EcogExtractHighGammaTrial(double(squeeze(data(:, iTrial, :))),...
            %                           fs, fDown, fSB, tw, gtw, normFactor,...
            %                           normType);
    end
end

ieegSpikeBand = ieegStructClass(ieegSBtemp, fDown, gtw, fSB, name);
ieegSBP = (squeeze(mean(log10(ieegSBtemp.^2), 3)));

end