
%%%%%
% Segmenting data in 30-s epochs and adding scoring labels 

%% Paths

clear;
close all;

addpath '/Users/nico/Documents/ICM'
run localdef.m

path_data = [path_data_Iceberg filesep 'healthy'];
addpath((path_fieldtrip));
addpath((path_fieldtrip_adv));

files=dir([path_data filesep '*.edf']);
ft_defaults;

%% loop on subjects

all_problematicFiles=[];
redo=1;

% for nF=1:length(files)
for nF=1
    
    % Which subject
    file_name = files(nF).name;
    folder_name = files(nF).folder;
    seps=findstr(file_name,'.');
    SubID=file_name(1:seps(1)-1);
    fprintf('... working on %s (%g/%g)\n',file_name,nF,length(files))
        
    % Read header
    hdr=ft_read_header([folder_name filesep file_name]);
    all_channels = string(hdr.label);

    % Define epochs
    cfg=[];
    cfg.trialfun                = 'ET_trialfun';
    cfg.SubID                   = SubID;
    cfg.dataset                 = [folder_name filesep file_name];
    cfg.trialdef.lengthSegments = 30; % in seconds;
    cfg = ft_definetrial(cfg);

    % Filtering
    cfg.channel        = cellstr(all_channels);
    cfg.demean         = 'yes';
    cfg.lpfilter       = 'yes';        % enable high-pass filtering
    cfg.lpfilttype     = 'but';
    cfg.lpfiltord      = 4;
    cfg.lpfreq         = 40;
    cfg.hpfilter       = 'yes';        % enable high-pass filtering
    cfg.hpfilttype     = 'but';
    cfg.hpfiltord      = 4;
    cfg.hpfreq         = 0.1;
    cfg.dftfilter      = 'yes';        % enable notch filtering to eliminate power line noise
    cfg.dftfreq        = [50 100 150]; % set up the frequencies for notch filtering
    
    % Re-referencing to linked mastoids
    cfg.reref = 'yes';
    cfg.refmethod = 'avg';
    cfg.refchannel = {'A1', 'A2'}; % linked mastoids

    % Preprocessing
    dat                = ft_preprocessing(cfg); 

    % Resampling
    cfgbs=[];
    cfgbs.resamplefs   = 256;
    cfgbs.detrend      = 'no';
    cfgbs.demean       = 'yes';
    data               = ft_resampledata(cfgbs,dat); 
    
    % Add scoring labels to each 30s epoch
    addpath([path_data filesep 'Hypno'])
    ScoringLabels = readcell([path_data filesep 'Hypno' filesep sprintf('%sS.TXT',SubID)]);
    cfg.ScoringLabels = ScoringLabels;
    %ft_databrowser(cfg, data)
    
    % Save the structure
    % save([path_data filesep 'f_ft_' file_name(1:end-4)],'data');
    
    
end




