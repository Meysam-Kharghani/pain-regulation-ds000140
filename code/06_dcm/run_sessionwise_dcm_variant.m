function run_sessionwise_dcm_variant(variant, mode)
%RUN_SESSIONWISE_DCM_VARIANT Run one DCM variant in smoke-test or full mode.
%
% This function reuses the session-wise, rating-adjusted, all-passive DCM
% pipeline that was successfully used for the primary and bilateral models.
% Every run is estimated as a separate DCM. In full mode, participant-level
% and original hierarchical PEBs are generated first; low-collinearity
% reduced group PEBs are then fitted with run_reduced_group_peb_sensitivity.
%
% Modes:
%   smoke   - estimate and deeply audit the first participant (9 DCMs)
%   full    - estimate/resume all 288 DCMs, audit them, then run reduced PEB
%   reduced - rerun only the reduced group PEBs from existing subject PEBs
%   audit   - audit the existing smoke or full output automatically

if nargin < 1 || isempty(variant)
    error('A DCM variant is required.');
end
if nargin < 2 || isempty(mode)
    mode = 'smoke';
end

variant = char(string(variant));
mode = lower(char(string(mode)));

allowed_variants = { ...
    'control_value_descending_primary', ...
    'control_value_bilateral_sensitivity', ...
    'descending_affective_amygdala_left', ...
    'descending_affective_amygdala_right', ...
    'sensory_interoceptive_contralateral_right', ...
    'sensory_interoceptive_ipsilateral_left'};
if ~ismember(variant, allowed_variants)
    error('Unsupported variant: %s', variant);
end

allowed_modes = {'smoke','full','reduced','audit'};
if ~ismember(mode, allowed_modes)
    error('Unknown mode %s. Use smoke, full, reduced, or audit.', mode);
end

root = getenv('DS000140_ROOT');
if isempty(root)
    error('DS000140_ROOT is not defined.');
end
if ~isfolder(root)
    error('DS000140_ROOT does not exist: %s', root);
end

required_functions = { ...
    'get_sessionwise_dcm_config', ...
    'run_sessionwise_dcm_pipeline', ...
    'run_reduced_group_peb_sensitivity', ...
    'validate_sessionwise_dcm_inputs', ...
    'audit_sessionwise_dcm_outputs'};
for i = 1:numel(required_functions)
    if exist(required_functions{i}, 'file') ~= 2
        error('Required function is not on the MATLAB path: %s', ...
            required_functions{i});
    end
end

if exist('spm', 'file') ~= 2
    error('SPM is not on the MATLAB path.');
end
[spm_name, spm_release] = spm('Ver');
if ~contains(lower([char(spm_name) ' ' char(spm_release) ' ' spm('Dir')]), 'spm25')
    error('SPM25 is required; detected %s %s at %s.', ...
        spm_name, spm_release, spm('Dir'));
end

fprintf('\n============================================================\n');
fprintf('Variant: %s\n', variant);
fprintf('Mode: %s\n', mode);
fprintf('SPM: %s %s\n', spm_name, spm_release);
fprintf('Dataset: %s\n', root);
fprintf('============================================================\n\n');

validate_sessionwise_dcm_inputs(variant);

switch mode
    case 'smoke'
        cfg = get_sessionwise_dcm_config(variant);
        cfg.max_subjects = 1;
        cfg.estimate_dcm = true;
        cfg.run_all_condition_hierarchical_peb = false;
        cfg.run_direction_hierarchical_peb = false;
        % Re-estimate the nine smoke-test DCMs so stale partial estimates
        % cannot be silently reused.
        cfg.overwrite = true;

        run_sessionwise_dcm_pipeline(cfg);
        audit_sessionwise_dcm_outputs(variant, 'smoke');

        fprintf('\nSMOKE TEST PASSED: %s\n', variant);

    case 'full'
        cfg = get_sessionwise_dcm_config(variant);
        cfg.max_subjects = 0;
        cfg.estimate_dcm = true;
        cfg.run_all_condition_hierarchical_peb = true;
        cfg.run_direction_hierarchical_peb = true;
        % Resume safely and reuse successful smoke-test estimates.
        cfg.overwrite = false;

        run_sessionwise_dcm_pipeline(cfg);
        audit_sessionwise_dcm_outputs(variant, 'full');

        % The original hierarchical PEB is needed to create the 33
        % participant-level PEBs. Scientific interpretation should use the
        % reduced low-collinearity designs written to tables/reduced_peb.
        run_reduced_group_peb_sensitivity(variant);

        fprintf('\nFULL DCM + REDUCED PEB PASSED: %s\n', variant);

    case 'reduced'
        audit_sessionwise_dcm_outputs(variant, 'full');
        run_reduced_group_peb_sensitivity(variant);
        fprintf('\nREDUCED PEB RERUN PASSED: %s\n', variant);

    case 'audit'
        audit_sessionwise_dcm_outputs(variant, 'auto');
end
end
