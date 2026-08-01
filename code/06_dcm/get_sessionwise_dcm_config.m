function cfg = get_sessionwise_dcm_config(variant)
%GET_SESSIONWISE_DCM_CONFIG Return the merged configuration for one run-separated DCM variant.
%
% Primary session-aware DCM sensitivity analysis for ds000140:
%   - every QC-retained run is estimated independently;
%   - all available passive runs are retained;
%   - each participant contributes exactly one up and one down run;
%   - participant-level PEBs estimate Up-minus-Passive and
%     Down-minus-Passive while adjusting for linear run position;
%   - a separate participant-level PEB estimates Up-minus-Down.
%
% In the archived QC-retained inventory this corresponds to:
%   288 independent run DCMs across 33 participants
%   222 passive runs
%    33 up-regulation runs
%    33 down-regulation runs
%
% Passive availability varies after QC (4-7 passive runs per participant),
% so no participant is discarded merely for lacking one passive session.

if nargin < 1 || isempty(variant)
    variant = 'control_value_descending_primary';
end

cfg = get_sessionwise_dcm_variant_config(variant);

cfg.inventory_tsv = fullfile( ...
    cfg.derivatives_dir, 'first_level_inputs', 'tables', ...
    'first_level_run_inventory.tsv');
cfg.roi_definitions_tsv = fullfile( ...
    cfg.derivatives_dir, 'roi_masks', 'roi_definitions.tsv');
cfg.behavior_subjects_tsv = fullfile( ...
    cfg.derivatives_dir, 'beh', 'behavior_models', 'tables', ...
    'subject_level_behavior_parameters.tsv');
cfg.resampled_mask_cache = fullfile( ...
    cfg.derivatives_dir, 'roi_masks', 'resampled_to_bold');

cfg.analysis_name = 'sessionwise_rating_adjusted';
cfg.outdir = [cfg.outdir '_sessionwise_rating_adjusted'];

cfg.runs_to_include = 1:9;
cfg.passive_policy = 'all_available_QC_retained_passive_runs';
cfg.adjust_within_subject_run_position = true;

cfg.expected_total_subjects = 33;
cfg.expected_n_dcm_runs = 288;
cfg.expected_n_passive_runs = 222;
cfg.expected_n_up_runs = 33;
cfg.expected_n_down_runs = 33;
cfg.min_passive_runs_per_subject = 4;
cfg.max_passive_runs_per_subject = 7;
cfg.require_expected_n_dcm_runs = true;
cfg.require_full_regulation_runs = true;

% PEB switches. Keep all false during the one-participant smoke test.
cfg.run_all_condition_hierarchical_peb = true;
cfg.run_direction_hierarchical_peb = true;

% Group-level covariates. The all-condition PEB contains the second-level
% effects PassiveAtMeanRun, UpMinusPassive, DownMinusPassive and RunPosition.
cfg.peb_covariates_all_conditions = { ...
    'reg_success_z_up', ...
    'reg_success_z_down', ...
    'passive_mean_rating_z', ...
    'passive_sensory_slope_z', ...
    'up_first' ...
};

% The direction PEB contains DirectionMean and UpMinusDown.
cfg.peb_covariates_direction = { ...
    'reg_success_z_up', ...
    'reg_success_z_down', ...
    'passive_mean_rating_z', ...
    'passive_sensory_slope_z', ...
    'up_first' ...
};
end
