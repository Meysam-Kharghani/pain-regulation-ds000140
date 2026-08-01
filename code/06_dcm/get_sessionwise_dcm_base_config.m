function cfg = get_sessionwise_dcm_base_config()
%GET_SESSIONWISE_DCM_BASE_CONFIG Return shared configuration for run-separated DCM/PEB analyses.
%
% Set DS000140_ROOT and SPM25_PATH in the environment before running.
%
% Design choices:
% - Echo time is fixed to 0.020 s to match BIDS EchoTime.
% - DCM variants are focused five-node models to keep subject-level DCM stable.
% - Lateralized structures are handled with explicit primary/sensitivity variants,
%   not by silently assuming one hemisphere is irrelevant.
% - All experimental inputs directly enter all selected nodes via C, so B parameters
%   are interpreted as condition-specific modulation of connections rather than
%   omitted direct task responses.

% -------------------------
% Paths
% -------------------------
project_root = getenv('DS000140_ROOT');
if isempty(project_root)
    project_root = '<PROJECT_ROOT>';
end

deriv = fullfile(project_root, 'derivatives');

cfg = struct();
cfg.project_root = project_root;
cfg.derivatives_dir = deriv;

% Set this if SPM25 is not already on the MATLAB path.
spm_path = getenv('SPM25_PATH');
if isempty(spm_path)
    spm_path = '';
end
cfg.spm_path = spm_path;

cfg.inventory_tsv = fullfile(deriv, 'first_level_inputs', 'tables', 'first_level_run_inventory.tsv');
cfg.roi_definitions_tsv = fullfile(deriv, 'roi_masks', 'roi_definitions.tsv');
cfg.behavior_subjects_tsv = fullfile(deriv, 'beh', 'behavior_models', 'tables', 'subject_level_behavior_parameters.tsv');

% The DCM code first looks here for run-specific masks resampled to each BOLD grid.
% If no cached resampled mask exists, it falls back to the ROI mask path in roi_definitions.tsv.
cfg.resampled_mask_cache = fullfile(deriv, 'roi_masks', 'resampled_to_bold');

% -------------------------
% Run/data selection
% -------------------------
cfg.runs_to_include = 1:9;
cfg.max_subjects = 0;             % 0 = all subjects
cfg.estimate_dcm = true;
cfg.run_peb = true;
cfg.overwrite = false;
cfg.allow_missing_roi = false;

% Explicit QC-lock policy for DCM.
% Main DCM must use the same QC-retained run set as smoothing/GLM/trialwise analyses:
% 288 runs total, with all 66 regulation runs retained.
cfg.dcm_include_policy = 'main_keep';  % keep KEEP, KEEP_WITH_WARNING, KEEP_FOR_MAIN_BUT_SENSITIVITY_EXCLUDE_CANDIDATE
cfg.expected_n_dcm_runs = 288;
cfg.require_expected_n_dcm_runs = true;
cfg.require_full_regulation_runs = true;
cfg.expected_n_up_runs = 33;
cfg.expected_n_down_runs = 33;
cfg.expected_total_subjects = 33;

% Prefer unsmoothed fMRIPrep BOLD for small ROI/DCM analyses.
cfg.bold_column_preference = {'bold_unsmoothed','preproc_bold','bold_preproc','bold'};
cfg.events_column_preference = {'events_stim','events_long','events','events_tsv'};
cfg.confounds_column_preference = {'confounds_glm','confounds_tsv','confounds'};

% -------------------------
% DCM constants
% -------------------------
cfg.echo_time_sec = 0.020;        % BIDS EchoTime = 0.020 s
cfg.input_names = {'passive','up','down'};
cfg.min_event_duration_sec = 0.1;
cfg.use_full_A = false;

cfg.dcm_options = struct();
cfg.dcm_options.nonlinear = 0;
cfg.dcm_options.two_state = 0;
cfg.dcm_options.stochastic = 0;
cfg.dcm_options.induced = 0;
cfg.dcm_options.centre = 1;

cfg.peb_fields = {'B'};
cfg.peb_covariates = { ...
    'reg_success_z_up', ...
    'reg_success_z_down', ...
    'mean_reg_success_z', ...
    'passive_mean_rating_z', ...
    'passive_sensory_slope_z', ...
    'up_first' ...
};

% Model-specific fields are filled by get_sessionwise_dcm_variant_config.
cfg.model_variant = 'UNSPECIFIED';
cfg.model_family = 'UNSPECIFIED';
cfg.model_role = 'UNSPECIFIED';
cfg.rationale = '';
cfg.outdir = fullfile(deriv, 'dcm_unspecified');
cfg.roi_specs = struct('name', {}, 'pattern', {});
cfg.A_edges = {};
cfg.B_edges = {};
cfg.C_drives = {};
end
