%% Initialise paths and toolboxes
clear;
close all;
set(0,'DefaultUIControlFontSize',16);

if exist('ft_read_data.m')==0
    warning('You need to add fiedltrip to your path!')
    fprintf('>>> Select the fieldtrip main folder\n')
    ft_folder = uigetdir('','Select the fieldtrip main folder');
    addpath(ft_folder)
    ft_defaults;
end

% Path to EDF files: select folder containing the EDF files
fprintf('>>> Select the folder containing the EDF files\n')
subfolder = uigetdir('','Select the folder containing the EDF files');

%% Select EDFs to plot
% Return the subject IDs from the data folder
filelist = dir([subfolder filesep '**' filesep '*.mat']);
pick=listdlg('ListString',{filelist.name},'PromptString','Select the EDF file to check');
filelist = filelist(pick);
fprintf('>>> You have selected %g EEG file\n',length(filelist))

%% Select channels to plot
load([filelist.folder filesep filelist.name])

%% display data after preprocessing
cfg=[];
cfg.continuous      = 'no';
cfg.allowoverlap    = 'true';
cfg.viewmode        = 'vertical';
cfg.ylim            = 'maxmin';
cfg                 = ft_databrowser(cfg,data);


