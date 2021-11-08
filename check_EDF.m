

%% EDF format checks

% There are 3 potential issues related to the conversion of raw signals to
% EDF formats:

% 1. Signal clipping: signal cut once it exceeds an amplitude threshold
% (the min-max range set before edf conversion was too narrow)

% 2. Bit depth: signal shows a stair-like progression (the min-max
% range set before edf conversion was too wide)

% 3. Inverted polarity: signal multiplied by -1 

%% Load the data

clear;
close all;

% Healthy or Park subjects?
rootdir = '/Users/nico/Documents/ICM/Iceberg/Data/PARK';

addpath '/Users/nico/Documents/GitHub/Dream-Toolkit'
run localdef.m

Subs_healthy = {'73','75','91','94','98','99','104','106','107','109'};
Subs_parks = {'79','95','165','185','193','222','246','269','282','284'};

% Loop across subjects
for S = 1

    % Parameters subject
    subID = cell2mat(strcat(Subs_parks(S),'.edf'));
    separator = strfind(subID,'.edf'); 
    Sub = subID(1:separator(1)-1);

    % Import the data
    cfg = [];
    cfg.dataset = [rootdir filesep subID];
    
    fprintf(1,'Importing data from Subject %s...\n',Sub)
    data = ft_read_data(cfg.dataset);
    hdr = ft_read_header(cfg.dataset);
    
    all_channels = string(hdr.label);
    Num_ch = numel(all_channels);
     
%%%   Visualise the data
%     cfg             = [];
%     cfg.dataset     = [rootdir filesep subID];
%     fg.channel      = cellstr(Channels);
%     Preproc_data    = ft_preprocessing(cfg);
%     cfg.blocksize   = 30; % in sec
%     cfg.channel     = cellstr(Channels); 
%     cfg.viewmode    = 'vertical';
%     ft_databrowser(cfg, Preproc_data);
    
    
    %% Check for signal clipping and bit depth issue
        
    fprintf(1,'Plotting histograms for each of the %s channels...\n',string(Num_ch))

    for i = 1:Num_ch
        
        Data = data(i,:);

        f=figure;

        % Signal clipping: outstanding values (ie, min/max) in the histogram?
        subplot(1,2,1); ax=gca;

        histogram(Data,-max(abs(Data)):0.05:max(abs(Data)),'EdgeColor','#1167b1') 
        xlim([-1.05 1.05]*max(abs(Data)))
        t = title('Data points distribution');
        t.FontWeight = 'normal';
        xlabel('Amplitude'); ylabel('Distribution')
        ax.FontSize = 14;

        % Plot the absolute difference in amplitude between neihboring data points 
        % --> Signal clipping if two values peak out
        % --> Bit depth issue if gaps observed between evenly distributed values
        subplot(1,2,2); ax=gca;

        delta_ampl = abs(diff(Data));

        histogram(delta_ampl,0:0.01:50,'EdgeColor','#1167b1')

        t = title({'Absolute difference in amplitude between';'neighboring data points'});
        t.FontWeight = 'normal';
        xlabel('Delta amplitude'); ylabel('Distribution') 
        ax.FontSize = 14;
        f.Position = [459,1143,906,420];
        T = sgtitle({sprintf('Subject %s',Sub);sprintf('Channel %s',all_channels(i))}); 
        T.FontWeight = 'bold';
        f.Position = [-96,1387,906,420];

    end

    %% Check for polarity issue

    % Compute the correlation between the channels. Positive correlation
    % means that all channels have the same polarity (but doesn't rule out the
    % possibilty that all have an inverse polarity——needs to be manually
    % checked by comparing with raw signals on Compumedics)
    [r, pV] = corr(data');

    % Plot the correlation matrix
    g=figure; ax=gca;
    imagesc(r); 

    colorbar
    caxis([-1 1])
    xticks(1:Num_ch)
    yticks(1:Num_ch)
    xticklabels(all_channels)
    yticklabels(all_channels)
    ax.XAxis.TickLength = [0 0];
    ax.YAxis.TickLength = [0 0];
    xtickangle(ax,45)
    t = title({sprintf('Subject %s',Sub);'Correlation between channels'});
    t.FontWeight = 'normal';
    
    % Add text for negative correlation (r < -0.2)
    t = cell(Num_ch, Num_ch);
    
    for i=1:Num_ch
        for j=1:Num_ch
            t(i, j) = cellstr(num2str(round(r(i,j), 2)));
        end
    end
    
    t = cellfun(@str2num,t);
    [x,y] = find(t<=-0.2);
    
    for i = 1:numel(x)
        text(x(i), y(i), string(t(x(i),y(i))), 'HorizontalAlignment', 'Center', 'FontSize', 12);
    end
    
    ax.FontSize = 14;
    g.Position = [811,1185,773,623];
    

end



