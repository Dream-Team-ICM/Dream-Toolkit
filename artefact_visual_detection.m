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
fprintf('>>> Select the folder containing the EEG files\n')
subfolder = uigetdir('','Select the folder containing the EDF files');

%% Select EEG files to plot
% Return the subject IDs from the data folder
filelist = dir([subfolder filesep '**' filesep '*.set']);
pick=listdlg('ListString',{filelist.name},'PromptString','Select the EDF file to check');
filelist = filelist(pick);
fprintf('>>> You have selected %g files\n',length(filelist))

%% Import FDT (EEGlab) data into fieldtrip structure
cfg=[];
cfg.dataset         = [filelist.folder filesep filelist.name];
preprocdata        = ft_preprocessing(cfg); % read raw data

%% reject trials
cfg          = [];
cfg.method   = 'summary';
cfg.alim     = 5e-5;
data        = ft_rejectvisual(cfg,oridata);

%% display data
cfg=[];
cfg.continuous='no';
cfg.allowoverlap='true';
cfg.viewmode='vertical';
cfg = ft_databrowser(cfg, oridata);


