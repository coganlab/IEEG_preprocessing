function ax = visTimePlot3(timeEpoch,signal2plot,options)
%VISTIMEPLOT Visualize channel averaged plot of the entire time series
%   Detailed explanation goes here
% timeEpoch - 3 dimensions of time points 
arguments
    timeEpoch double % 2D time values in seconds [-0.5000    2.0000;   -0.5000    1.0000;   -1.0000    1.5000]
    signal2plot double % 2D -  channels x timeseries
    options.colval = [1 0 1]; % color value
    options.fs = 200; % sampling frequency
    options.labels = {'Auditory','Go','ResponseOnset'}
    options.tileaxis = []
    options.ybound = 0;
end
fs = options.fs;
colval = options.colval;

% figure;
hold on;
if(isempty(options.tileaxis))
    t = tiledlayout(1,3,'TileSpacing','compact');
    ax1 =  axes(t);
    ax1.Layout.Tile = 1;
    ax2 =  axes(t);
    ax2.Layout.Tile = 2;
    ax3 =  axes(t);
    ax3.Layout.Tile = 3;
else
    %t = options.tileLayout;
    ax1 = options.tileaxis{1};
    ax2 = options.tileaxis{2};
    ax3 = options.tileaxis{3};
end
    sig1M=mean(signal2plot); % extract mean
    sig1S=std(signal2plot)./sqrt(size(signal2plot,1)); % extract standard error
   timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2),(timeEpoch(1,2)-timeEpoch(1,1))*fs );
   timeGamma2 = linspace(timeEpoch(2,1),timeEpoch(2,2),(timeEpoch(2,2)-timeEpoch(2,1))*fs);
   timeGamma3 = linspace(timeEpoch(3,1),timeEpoch(3,2),(timeEpoch(3,2)-timeEpoch(3,1))*fs );

    
    %
    hold(ax1,'on')
    sig1M2plot = sig1M(1:round(length(timeGamma1)));
    sig1S2plot = sig1S(1:round(length(timeGamma1)));
    h = plot(ax1,timeGamma1,sig1M2plot,'LineWidth',2,'Color',colval);
    
    h = patch(ax1,[timeGamma1,timeGamma1(end:-1:1)],[sig1M2plot + sig1S2plot, ...
    sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
    set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    hold on;
    xline(ax1,timeGamma1(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGamma1(1) timeGamma1(end)])
    yline(ax1,options.ybound, ':','','LineWidth',1, 'Color','k');
    
    xlabel(ax1,options.labels{1})
    
     formatTicks(ax1)
     
    

    
    %ax2.Layout.Tile = 2;
    hold(ax2,'on')
    startTimePoint = round(length(timeGamma1))+1;
    sig1M2plot = sig1M(startTimePoint:startTimePoint+length(timeGamma2)-1);
    sig1S2plot = sig1S(startTimePoint:startTimePoint+length(timeGamma2)-1);
    h = plot(ax2,timeGamma2,sig1M2plot,'LineWidth',2,'Color',colval);
    h = patch(ax2,[timeGamma2,timeGamma2(end:-1:1)],[sig1M2plot + sig1S2plot, ...
    sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
    set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    xline(ax2,timeGamma2(1),':');
    xline(ax2,timeGamma2(end),':');
    ax2.YAxis.Visible = 'off';
    ax2.Box = 'off';
    xlim(ax2,[timeGamma2(1) timeGamma2(end)])
    yline(ax2,options.ybound, ':','','LineWidth',1, 'Color','k');
  
    
    xlabel(ax2,options.labels{2})
    
    
   formatTicks(ax2)
     
    
    %ax3.Layout.Tile = 3;
    hold(ax3,'on')
    startTimePoint = startTimePoint+round(length(timeGamma2));
    sig1M2plot = sig1M(startTimePoint:end);
    sig1S2plot = sig1S(startTimePoint:end);
    h = plot(ax3,timeGamma3,sig1M2plot,'LineWidth',2,'Color',colval);
     h = patch(ax3,[timeGamma3,timeGamma3(end:-1:1)],[sig1M2plot + sig1S2plot, ...
    sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
     set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    xline(ax3,timeGamma3(1),':');
    xline(ax3,timeGamma3(end),':');
    ax3.YAxis.Visible = 'off';
    ax3.Box = 'off';
    xlim(ax3,[timeGamma3(1) timeGamma3(end)])
    yline(ax3,options.ybound, ':','','LineWidth',1, 'Color','k');

    
    xlabel(ax3,options.labels{3})
    
    formatTicks(ax3)
     
     ax{1} = ax1;
    ax{2} = ax2;
    ax{3} = ax3;
% Link the axes
    linkaxes([ax1 ax2 ax3], 'y')
    
end

