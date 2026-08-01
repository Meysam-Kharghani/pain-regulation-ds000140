function run_sessionwise_dcm_pipeline(cfg)
%RUN_SESSIONWISE_DCM_PIPELINE Estimate run-level DCMs and hierarchical PEB models.
%
% The primary model uses every available QC-retained passive run plus the
% single up- and down-regulation runs for each participant. Every run is an
% independent DCM, so neuronal and haemodynamic states reset at run
% boundaries. Rating periods, a 128-s DCT high-pass basis and standardized
% confounds enter DCM.Y.X0.
%
% Participant-level PEBs use all available sessions and treatment coding:
%   PassiveAtMeanRun, UpMinusPassive, DownMinusPassive, RunPosition.
% A separate two-session participant-level PEB estimates UpMinusDown.
%
% This remains a sensitivity analysis and does not turn the exploratory
% coupling result into independent confirmation.

if nargin < 1 || isempty(cfg)
    cfg = get_sessionwise_dcm_config( ...
        'control_value_descending_primary');
end

setup_spm25(cfg);
ensure_dir(cfg.outdir);
for name = {'dcm_structs','estimated_dcms','peb_group','peb_subjects', ...
        'tables','cache_unzipped'}
    ensure_dir(fullfile(cfg.outdir, name{1}));
end

diary_file = fullfile(cfg.outdir, 'sessionwise_dcm_peb_run_log.txt');
if exist(diary_file, 'file') && cfg.overwrite
    delete(diary_file);
end
diary(diary_file);
cleanup_diary = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('Started: %s\n', datestr(now, 30));
fprintf('Variant: %s\n', cfg.model_variant);
fprintf('Passive policy: %s\n', cfg.passive_policy);
fprintf('MATLAB: %s (%s)\n', version, version('-release'));
[spm_name, spm_release] = spm('Ver');
fprintf('SPM: %s %s\n', spm_name, spm_release);
fprintf('SPM directory: %s\n', spm('Dir'));

inv = readtable(cfg.inventory_tsv, 'FileType', 'text', 'Delimiter', '\t');
inv = normalize_subject_column(inv);
inv = filter_inventory(inv, cfg);
beh = readtable(cfg.behavior_subjects_tsv, 'FileType', 'text', 'Delimiter', '\t');
beh = normalize_subject_column(beh);
roi_defs = readtable(cfg.roi_definitions_tsv, 'FileType', 'text', 'Delimiter', '\t');
roi_selection = select_rois(roi_defs, cfg);
writetable(roi_selection, fullfile(cfg.outdir, 'tables', 'dcm_roi_selection.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

subjects = unique(inv.subject, 'stable');
if isfield(cfg, 'max_subjects') && cfg.max_subjects > 0
    subjects = subjects(1:min(cfg.max_subjects, numel(subjects)));
end
inv = inv(ismember(inv.subject, subjects), :);

summary_rows = {};
records = struct('subject', {}, 'run', {}, 'condition', {}, 'path', {});

for si = 1:numel(subjects)
    sub = subjects{si};
    rows = inv(strcmp(inv.subject, sub), :);
    runs = numeric_run_vector(rows.run);
    [runs, order] = sort(runs);
    rows = rows(order, :);

    for ri = 1:height(rows)
        run_num = runs(ri);
        fprintf('\n--- %s run-%02d ---\n', sub, run_num);
        try
            [DCM, condition, run_meta] = build_run_dcm( ...
                rows(ri,:), sub, run_num, roi_selection, cfg);
            struct_path = fullfile(cfg.outdir, 'dcm_structs', ...
                sprintf('dcm_%s_run-%02d_%s_unestimated.mat', ...
                sub, run_num, condition));
            save(struct_path, 'DCM', '-v7.3');

            est_path = fullfile(cfg.outdir, 'estimated_dcms', ...
                sprintf('dcm_%s_run-%02d_%s_estimated.mat', ...
                sub, run_num, condition));
            if cfg.estimate_dcm
                if exist(est_path, 'file') && ~cfg.overwrite
                    fprintf('Using existing estimate: %s\n', est_path);
                    loaded = load(est_path, 'DCM');
                    DCM_est = loaded.DCM; %#ok<NASGU>
                else
                    fprintf('Estimating run-specific DCM...\n');
                    DCM_est = spm_dcm_estimate(DCM);
                    DCM = DCM_est; %#ok<NASGU>
                    save(est_path, 'DCM', '-v7.3');
                end
                rec.subject = sub;
                rec.run = run_num;
                rec.condition = condition;
                rec.path = est_path;
                records(end+1) = rec; %#ok<AGROW>
            end

            summary_rows(end+1,:) = {sub, run_num, condition, 'OK', '', ...
                run_meta.n_scans, height(roi_selection), ...
                run_meta.n_stim_events, run_meta.n_rating_events, ...
                run_meta.n_confounds, struct_path, est_path}; %#ok<AGROW>
        catch ME
            fprintf(2, 'FAILED %s run-%02d: %s\n', sub, run_num, ME.message);
            summary_rows(end+1,:) = {sub, run_num, '', 'FAILED', ...
                ME.message, NaN, height(roi_selection), NaN, NaN, NaN, ...
                '', ''}; %#ok<AGROW>
        end
    end
end

summary = cell2table(summary_rows, 'VariableNames', { ...
    'subject','run','condition','status','error','n_scans','n_regions', ...
    'n_stim_events','n_rating_events','n_confounds', ...
    'dcm_struct_path','estimated_dcm_path'});
summary = sortrows(summary, {'subject','run'});
writetable(summary, fullfile(cfg.outdir, 'tables', ...
    'sessionwise_dcm_run_summary.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

if any(~strcmp(summary.status, 'OK'))
    error(['One or more run-specific DCMs failed. PEB was not run. ', ...
        'Inspect sessionwise_dcm_run_summary.tsv.']);
end
validate_complete_records(records, subjects, inv, cfg);

record_table = struct2table(records);
record_table = sortrows(record_table, {'subject','run'});
writetable(record_table, fullfile(cfg.outdir, 'tables', ...
    'sessionwise_dcm_file_index.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');
export_run_level_B(records, cfg, fullfile(cfg.outdir, 'tables', ...
    'sessionwise_run_b_parameters.tsv'));

if isfield(cfg, 'run_all_condition_hierarchical_peb') && ...
        cfg.run_all_condition_hierarchical_peb
    run_all_condition_hierarchical_peb(records, subjects, beh, cfg);
end
if isfield(cfg, 'run_direction_hierarchical_peb') && ...
        cfg.run_direction_hierarchical_peb
    run_direction_hierarchical_peb(records, subjects, beh, cfg);
end

condition_counts = count_conditions(records);
manifest = table(string(cfg.model_variant), string(cfg.analysis_name), ...
    string(cfg.passive_policy), numel(subjects), numel(records), ...
    condition_counts.passive, condition_counts.up, condition_counts.down, ...
    cfg.include_rating_nuisance, cfg.high_pass_sec, ...
    cfg.microtime_resolution, cfg.microtime_onset, ...
    'VariableNames', {'model_variant','analysis_name','passive_policy', ...
    'n_subjects','n_run_dcms','n_passive_dcms','n_up_dcms','n_down_dcms', ...
    'rating_nuisance','high_pass_sec','microtime_resolution', ...
    'microtime_onset'});
writetable(manifest, fullfile(cfg.outdir, 'tables', ...
    'sessionwise_analysis_manifest.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

fprintf('\nFinished: %s\n', datestr(now, 30));
end

% =========================================================================
% Setup and validation
% =========================================================================
function setup_spm25(cfg)
if isfield(cfg, 'spm_path') && ~isempty(cfg.spm_path)
    addpath(cfg.spm_path);
end
if exist('spm', 'file') ~= 2
    error('SPM is not on the MATLAB path.');
end
spm('defaults', 'fmri');
[spm_name, spm_release] = spm('Ver');
version_text = lower([char(spm_name) ' ' char(spm_release) ' ' spm('Dir')]);
if ~contains(version_text, 'spm25')
    error('SPM25 is required, but detected: %s %s at %s', ...
        spm_name, spm_release, spm('Dir'));
end
end

function ensure_dir(path)
if ~exist(path, 'dir')
    mkdir(path);
end
end

function inv = filter_inventory(inv, cfg)
if ~ismember('run', inv.Properties.VariableNames)
    error('Inventory must contain a run column.');
end
runs = numeric_run_vector(inv.run);
keep = ismember(runs, cfg.runs_to_include);
vars = inv.Properties.VariableNames;
if ismember('status', vars)
    status = upper(strtrim(string(inv.status)));
    if any(status == "READY")
        keep = keep & status == "READY";
    elseif any(status == "OK")
        keep = keep & status == "OK";
    else
        keep = keep & ~contains(status, "FAIL");
    end
end
if ismember('analysis_decision', vars)
    decision = upper(strtrim(string(inv.analysis_decision)));
    keep = keep & startsWith(decision, "KEEP");
elseif ismember('keep_main', vars)
    keep = keep & bool_column(inv.keep_main);
end
inv = inv(keep, :);

keys = strcat(string(inv.subject), "_run-", compose('%02d', numeric_run_vector(inv.run)));
if numel(unique(keys)) ~= numel(keys)
    error('Inventory contains duplicate subject-run rows after QC filtering.');
end

if height(inv) ~= cfg.expected_n_dcm_runs
    error('Expected %d QC-retained runs; inventory filter returned %d.', ...
        cfg.expected_n_dcm_runs, height(inv));
end
if numel(unique(inv.subject)) ~= cfg.expected_total_subjects
    error('Expected %d subjects; inventory contains %d.', ...
        cfg.expected_total_subjects, numel(unique(inv.subject)));
end
end

function validate_complete_records(records, subjects, inv, cfg)
T = struct2table(records);
expected_selected = height(inv);
if numel(records) ~= expected_selected
    error('Expected %d selected run DCMs, found %d.', ...
        expected_selected, numel(records));
end

for i = 1:numel(subjects)
    sub = subjects{i};
    S = sortrows(T(strcmp(T.subject, sub), :), 'run');
    if height(S) ~= sum(strcmp(inv.subject, sub))
        error('Record count mismatch for %s.', sub);
    end
    if sum(strcmp(S.condition, 'up')) ~= 1 || ...
            sum(strcmp(S.condition, 'down')) ~= 1
        error('Subject %s must have exactly one up and one down DCM.', sub);
    end
    n_passive = sum(strcmp(S.condition, 'passive'));
    if n_passive < cfg.min_passive_runs_per_subject || ...
            n_passive > cfg.max_passive_runs_per_subject
        error('Subject %s has %d passive DCMs; expected %d-%d.', ...
            sub, n_passive, cfg.min_passive_runs_per_subject, ...
            cfg.max_passive_runs_per_subject);
    end
end

if ~isfield(cfg, 'max_subjects') || cfg.max_subjects == 0
    counts = count_conditions(records);
    if numel(records) ~= cfg.expected_n_dcm_runs || ...
            counts.passive ~= cfg.expected_n_passive_runs || ...
            counts.up ~= cfg.expected_n_up_runs || ...
            counts.down ~= cfg.expected_n_down_runs
        error(['Full-run counts mismatch. Found total=%d, passive=%d, ', ...
            'up=%d, down=%d.'], numel(records), counts.passive, ...
            counts.up, counts.down);
    end
end
end

function counts = count_conditions(records)
conditions = string({records.condition});
counts = struct();
counts.passive = sum(conditions == "passive");
counts.up = sum(conditions == "up");
counts.down = sum(conditions == "down");
end

function roi_selection = select_rois(roi_defs, cfg)
vars = roi_defs.Properties.VariableNames;
name_col = find_column(vars, {'roi','roi_name','name','label'});
mask_col = find_column(vars, {'mask_path','mask','path','file'});
if isempty(name_col) || isempty(mask_col)
    error('Could not identify ROI name and mask columns.');
end
rows = {};
for i = 1:numel(cfg.roi_specs)
    pattern = cfg.roi_specs(i).pattern;
    names = string(roi_defs.(name_col));
    hit = find(~cellfun(@isempty, regexp(cellstr(names), pattern, 'once')));
    if numel(hit) ~= 1
        error('ROI pattern %s matched %d rows.', pattern, numel(hit));
    end
    mask_path = resolve_project_path(char(string(roi_defs.(mask_col)(hit))), cfg.project_root);
    if ~exist(mask_path, 'file')
        error('ROI mask does not exist: %s', mask_path);
    end
    rows(end+1,:) = {cfg.roi_specs(i).name, char(names(hit)), ...
        sanitize_name(char(names(hit))), mask_path}; %#ok<AGROW>
end
roi_selection = cell2table(rows, 'VariableNames', {'dcm_name','roi','roi_safe','mask_path'});
end

% =========================================================================
% Run DCM construction
% =========================================================================
function [DCM, condition, meta] = build_run_dcm(row, sub, run_num, roi_selection, cfg)
bold_path = infer_existing_path(row, cfg.bold_column_preference, cfg.project_root, 'BOLD');
events_path = infer_existing_path(row, cfg.events_column_preference, cfg.project_root, 'events');
conf_path = infer_optional_path(row, cfg.confounds_column_preference, cfg.project_root);

bold_path = ensure_unzipped_nii(bold_path, fullfile(cfg.outdir, 'cache_unzipped', sub, sprintf('run-%02d', run_num)));
V = spm_vol(bold_path);
nscan = numel(V);
TR = get_TR(row, V);
[stim_events, rating_events, condition] = read_run_events(events_path);

micro_dt = TR / cfg.microtime_resolution;
n_micro = nscan * cfg.microtime_resolution;
heat = events_to_microtime_boxcar(stim_events, n_micro, micro_dt);
rating_micro = events_to_microtime_boxcar(rating_events, n_micro, micro_dt);
if cfg.include_rating_nuisance
    hrf = spm_hrf(micro_dt);
    rating_conv = conv(rating_micro, hrf);
    rating_conv = rating_conv(1:n_micro);
    rating_hrf = sample_microtime_to_scans( ...
        rating_conv, nscan, cfg.microtime_resolution, cfg.microtime_onset);
    rating_hrf = standardize_column(rating_hrf);
else
    rating_hrf = [];
end

confounds = build_run_confounds(conf_path, nscan);
high_pass = build_high_pass_basis(nscan, TR, cfg.high_pass_sec);
X0 = [ones(nscan,1), high_pass, rating_hrf, confounds];
X0 = X0(:, all(isfinite(X0), 1));
X0 = X0(:, std(X0, 0, 1) > 1e-12 | abs(mean(X0,1)) > 1e-12);
X0 = orth(X0);

y = extract_roi_mean_timeseries(bold_path, sub, run_num, roi_selection, cfg);
for j = 1:size(y,2)
    y(:,j) = standardize_column(y(:,j));
end

n = size(y,2);
DCM = struct();
DCM.xY.Dfile = '';
DCM.n = n;
DCM.v = nscan;
DCM.Y.y = y;
DCM.Y.dt = TR;
DCM.Y.X0 = X0;
DCM.Y.name = roi_selection.dcm_name(:)';
DCM.Y.Q = spm_Ce(ones(1,n) * nscan);
DCM.U.u = heat;
DCM.U.name = {cfg.session_input_name};
DCM.U.dt = micro_dt;
DCM.delays = repmat(TR/2, n, 1);
DCM.TE = cfg.echo_time_sec;
DCM.a = build_A_matrix(roi_selection.dcm_name, cfg);
DCM.b = zeros(n,n,1);
B_edges = unique_structural_B_edges(cfg);
for i = 1:size(B_edges,1)
    src = find(strcmp(roi_selection.dcm_name, B_edges{i,1}));
    tgt = find(strcmp(roi_selection.dcm_name, B_edges{i,2}));
    if ~isempty(src) && ~isempty(tgt)
        DCM.b(tgt,src,1) = 1;
    end
end
DCM.c = ones(n,1);
DCM.d = zeros(n,n,0);
DCM.options = cfg.dcm_options;
DCM.name = sprintf('dcm_%s_run-%02d_%s_%s', sub, run_num, condition, cfg.model_variant);
DCM.meta.subject = sub;
DCM.meta.run = run_num;
DCM.meta.condition = condition;
DCM.meta.events_path = events_path;
DCM.meta.bold_path = bold_path;
DCM.meta.confounds_path = conf_path;
DCM.meta.rating_nuisance = cfg.include_rating_nuisance;
DCM.meta.high_pass_sec = cfg.high_pass_sec;
DCM.meta.microtime_resolution = cfg.microtime_resolution;
DCM.meta.microtime_onset = cfg.microtime_onset;
DCM.meta.model_variant = cfg.model_variant;
DCM.meta.roi_selection = roi_selection;

meta.n_scans = nscan;
meta.n_stim_events = height(stim_events);
meta.n_rating_events = height(rating_events);
meta.n_confounds = size(X0,2);
meta.microtime_dt = micro_dt;
end

function [stim, rating, condition] = read_run_events(path)
T = readtable(path, 'FileType', 'text', 'Delimiter', '\t');
if ~ismember('onset', T.Properties.VariableNames)
    error('Events table has no onset: %s', path);
end
if ~ismember('duration', T.Properties.VariableNames)
    T.duration = zeros(height(T),1);
end
phase = repmat("", height(T), 1);
if ismember('event_phase', T.Properties.VariableNames)
    phase = lower(string(T.event_phase));
end
trial_type = repmat("", height(T), 1);
if ismember('trial_type', T.Properties.VariableNames)
    trial_type = lower(string(T.trial_type));
end
is_rating = phase == "rating" | contains(trial_type, "rating") | contains(trial_type, "response");
is_stim = phase == "stim" | contains(trial_type, "stim") | contains(trial_type, "heat");
if ~any(is_stim)
    is_stim = ~is_rating;
end
stim = T(is_stim,:);
rating = T(is_rating,:);

if ismember('condition', stim.Properties.VariableNames)
    cond = lower(string(stim.condition));
else
    cond = repmat("", height(stim), 1);
    tt = lower(string(stim.trial_type));
    cond(contains(tt, "up")) = "up";
    cond(contains(tt, "down")) = "down";
    cond(contains(tt, "passive")) = "passive";
end
cond = unique(cond(cond == "passive" | cond == "up" | cond == "down"));
if numel(cond) ~= 1
    error(['Each run-specific DCM must contain exactly one condition ', ...
        '(passive, up, or down); found: %s'], strjoin(cellstr(cond), ', '));
end
condition = char(cond);
end

function boxcar = events_to_microtime_boxcar(events, n_micro, dt)
boxcar = zeros(n_micro,1);
if isempty(events)
    return;
end
onset = double(events.onset);
duration = double(events.duration);
duration(~isfinite(duration) | duration <= 0) = 0.1;
for i = 1:numel(onset)
    s0 = max(1, floor(onset(i)/dt) + 1);
    s1 = min(n_micro, ceil((onset(i)+duration(i))/dt));
    if s1 < s0
        s1 = s0;
    end
    boxcar(s0:s1) = 1;
end
end

function scan_values = sample_microtime_to_scans(micro_values, nscan, T, T0)
if T0 < 1 || T0 > T
    error('microtime_onset must be between 1 and microtime_resolution.');
end
indices = (0:nscan-1) * T + T0;
indices = min(indices, numel(micro_values));
scan_values = micro_values(indices(:));
end

function X = build_high_pass_basis(nscan, TR, cutoff)
if isempty(cutoff) || cutoff <= 0
    X = [];
    return;
end
n_basis = floor(2 * (nscan * TR) / cutoff + 1);
if n_basis <= 1
    X = [];
else
    C = spm_dctmtx(nscan, n_basis);
    X = C(:,2:end);
end
end

function X = build_run_confounds(path, nscan)
if isempty(path) || ~exist(path, 'file')
    X = [];
    return;
end
T = readtable(path, 'FileType', 'text', 'Delimiter', '\t');
X = [];
for i = 1:width(T)
    values = T{:,i};
    if isnumeric(values) || islogical(values)
        x = double(values(:));
    else
        x = str2double(string(values));
    end
    if numel(x) < nscan
        x(end+1:nscan,1) = 0;
    elseif numel(x) > nscan
        x = x(1:nscan);
    end
    x(~isfinite(x)) = 0;
    if std(x) > 1e-12
        X(:,end+1) = standardize_column(x); %#ok<AGROW>
    end
end
end

function y = extract_roi_mean_timeseries(bold_path, sub, run_num, roi_selection, cfg)
V = spm_vol(bold_path);
nscan = numel(V);
nroi = height(roi_selection);
y = nan(nscan, nroi);
indices = cell(nroi,1);
for r = 1:nroi
    roi_safe = char(roi_selection.roi_safe(r));
    original_mask = char(roi_selection.mask_path(r));
    mask_path = find_resampled_mask(sub, run_num, roi_safe, original_mask, cfg);
    mask_cache = fullfile(cfg.outdir, 'cache_unzipped', sub, sprintf('run-%02d', run_num), 'masks');
    mask_path = ensure_unzipped_nii(mask_path, mask_cache);
    mask_path = ensure_mask_on_bold_grid(mask_path, bold_path, mask_cache, roi_safe);
    Vm = spm_vol(mask_path);
    mask = spm_read_vols(Vm);
    indices{r} = find(isfinite(mask) & mask > 0);
    if isempty(indices{r})
        error('Empty mask for ROI %s.', roi_safe);
    end
end
for t = 1:nscan
    volume = spm_read_vols(V(t));
    for r = 1:nroi
        values = volume(indices{r});
        values = values(isfinite(values));
        if isempty(values)
            error('No finite BOLD values for ROI %s at scan %d.', ...
                char(roi_selection.roi_safe(r)), t);
        end
        y(t,r) = mean(values);
    end
end
end

function out = ensure_mask_on_bold_grid(mask_path, bold_path, cache_dir, roi_safe)
Vref = spm_vol([bold_path ',1']);
Vmask = spm_vol(mask_path);
same_dim = isequal(Vref.dim, Vmask.dim);
same_affine = max(abs(Vref.mat(:) - Vmask.mat(:))) < 1e-4;
if same_dim && same_affine
    out = mask_path;
    return;
end
ensure_dir(cache_dir);
mask_copy = fullfile(cache_dir, [roi_safe '_source_mask.nii']);
if ~exist(mask_copy, 'file')
    copyfile(mask_path, mask_copy);
end
flags = struct('interp', 0, 'wrap', [0 0 0], 'mask', 0, 'mean', 0, ...
    'which', 1, 'prefix', 'r');
spm_reslice({[bold_path ',1']; mask_copy}, flags);
[folder, name, ext] = fileparts(mask_copy);
out = fullfile(folder, ['r' name ext]);
if ~exist(out, 'file')
    error('SPM failed to reslice mask %s to the BOLD grid.', mask_path);
end
end

function path = find_resampled_mask(sub, run_num, roi_safe, original_mask, cfg)
cache_dir = fullfile(cfg.resampled_mask_cache, sub, sprintf('run-%02d', run_num));
if exist(cache_dir, 'dir')
    hits = dir(fullfile(cache_dir, ['*' roi_safe '*resampled_to_bold*.nii*']));
    if isempty(hits)
        hits = dir(fullfile(cache_dir, ['*' roi_safe '*.nii*']));
    end
    if ~isempty(hits)
        path = fullfile(hits(1).folder, hits(1).name);
        return;
    end
end
path = original_mask;
end

function A = build_A_matrix(names, cfg)
n = numel(names);
A = eye(n);
if cfg.use_full_A
    A(:) = 1;
    return;
end
for i = 1:size(cfg.A_edges,1)
    src = find(strcmp(names, cfg.A_edges{i,1}));
    tgt = find(strcmp(names, cfg.A_edges{i,2}));
    if ~isempty(src) && ~isempty(tgt)
        A(tgt,src) = 1;
    end
end
end

function edges = unique_structural_B_edges(cfg)
edges = {};
for i = 1:size(cfg.B_edges,1)
    pair = cfg.B_edges(i,2:3);
    already_present = false;
    for j = 1:size(edges,1)
        if strcmp(edges{j,1}, pair{1}) && strcmp(edges{j,2}, pair{2})
            already_present = true;
            break;
        end
    end
    if ~already_present
        edges(end+1,:) = pair; %#ok<AGROW>
    end
end
end

% =========================================================================
% PEB analyses
% =========================================================================
function run_all_condition_hierarchical_peb(records, subjects, beh, cfg)
T = struct2table(records);
PEBs = cell(numel(subjects),1);
subject_summary = {};
design_rows = {};
field = {'B'};
run_scale = std((1:9), 0);

for i = 1:numel(subjects)
    sub = subjects{i};
    S = sortrows(T(strcmp(T.subject, sub), :), 'run');
    cond = string(S.condition);
    runs = double(S.run(:));
    if sum(cond == "up") ~= 1 || sum(cond == "down") ~= 1 || ...
            sum(cond == "passive") < cfg.min_passive_runs_per_subject
        error('Invalid all-condition session set for %s.', sub);
    end

    run_position = (runs - mean(runs)) / run_scale;
    X = [ones(height(S),1), double(cond == "up"), ...
        double(cond == "down"), run_position];
    if rank(X) < size(X,2)
        error('All-condition PEB design is rank deficient for %s.', sub);
    end

    GCM2 = to_cellstr_column(S.path);
    M2 = struct();
    M2.X = X;
    M2.Xnames = {'PassiveAtMeanRun','UpMinusPassive', ...
        'DownMinusPassive','RunPosition'};
    M2.Q = 'single';
    PEB2 = spm_dcm_peb(GCM2, M2, field);
    PEBs{i} = PEB2;
    save(fullfile(cfg.outdir, 'peb_subjects', ...
        sprintf('peb_%s_all_conditions.mat', sub)), ...
        'PEB2', 'GCM2', 'M2', 'field', '-v7.3');

    subject_summary(end+1,:) = {sub, height(S), sum(cond == "passive"), ...
        table_text(S(cond == "up",:), 'path', 1), ...
        table_text(S(cond == "down",:), 'path', 1), PEB2.F}; %#ok<AGROW>
    for j = 1:height(S)
        design_rows(end+1,:) = {sub, runs(j), char(cond(j)), ...
            X(j,1), X(j,2), X(j,3), X(j,4), ...
            table_text(S, 'path', j)}; %#ok<AGROW>
    end
end

D = cell2table(design_rows, 'VariableNames', {'subject','run','condition', ...
    'PassiveAtMeanRun','UpMinusPassive','DownMinusPassive', ...
    'RunPosition','dcm_path'});
writetable(D, fullfile(cfg.outdir, 'tables', ...
    'within_subject_all_condition_design.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

Sidx = cell2table(subject_summary, 'VariableNames', {'subject', ...
    'n_sessions','n_passive_sessions','up_dcm','down_dcm', ...
    'subject_peb_free_energy'});
writetable(Sidx, fullfile(cfg.outdir, 'tables', ...
    'all_condition_subject_peb_index.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

[M3, design3] = build_peb_design(subjects, beh, ...
    cfg.peb_covariates_all_conditions);
writetable(design3, fullfile(cfg.outdir, 'tables', ...
    'peb_design_all_conditions.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

PEB3 = spm_dcm_peb(PEBs, M3, field);
save(fullfile(cfg.outdir, 'peb_group', ...
    'peb_hierarchical_all_conditions_b.mat'), ...
    'PEB3', 'PEBs', 'M3', 'field', '-v7.3');
try
    [BMA3, BMR3] = spm_dcm_peb_bmc(PEB3);
    save(fullfile(cfg.outdir, 'peb_group', ...
        'bma_bmr_hierarchical_all_conditions_b.mat'), ...
        'BMA3', 'BMR3', 'PEB3', 'PEBs', 'M3', 'field', '-v7.3');
    export_peb_table(BMA3, fullfile(cfg.outdir, 'tables', ...
        'peb_bma_hierarchical_all_conditions_parameters.tsv'));
catch ME
    warning('All-condition hierarchical BMC failed: %s', ME.message);
    export_peb_table(PEB3, fullfile(cfg.outdir, 'tables', ...
        'peb_hierarchical_all_conditions_parameters.tsv'));
end
end

function run_direction_hierarchical_peb(records, subjects, beh, cfg)
T = struct2table(records);
PEBs = cell(numel(subjects),1);
subject_summary = {};
field = {'B'};
for i = 1:numel(subjects)
    sub = subjects{i};
    down = T(strcmp(T.subject, sub) & strcmp(T.condition, 'down'), :);
    up = T(strcmp(T.subject, sub) & strcmp(T.condition, 'up'), :);
    if height(down) ~= 1 || height(up) ~= 1
        error('Subject %s lacks exactly one up/down DCM.', sub);
    end
    down_path = table_text(down, 'path', 1);
    up_path = table_text(up, 'path', 1);
    GCM2 = {down_path; up_path};
    M2 = struct();
    M2.X = [1 -0.5; 1 0.5];
    M2.Xnames = {'DirectionMean','UpMinusDown'};
    M2.Q = 'single';
    PEB2 = spm_dcm_peb(GCM2, M2, field);
    PEBs{i} = PEB2;
    save(fullfile(cfg.outdir, 'peb_subjects', ...
        sprintf('peb_%s_up_minus_down.mat', sub)), ...
        'PEB2', 'GCM2', 'M2', 'field', '-v7.3');
    subject_summary(end+1,:) = {sub, down_path, up_path, PEB2.F}; %#ok<AGROW>
end

S = cell2table(subject_summary, 'VariableNames', ...
    {'subject','down_dcm','up_dcm','subject_peb_free_energy'});
writetable(S, fullfile(cfg.outdir, 'tables', ...
    'direction_subject_peb_index.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

[M3, design3] = build_peb_design(subjects, beh, ...
    cfg.peb_covariates_direction);
writetable(design3, fullfile(cfg.outdir, 'tables', ...
    'peb_design_direction.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');
PEB3 = spm_dcm_peb(PEBs, M3, field);
save(fullfile(cfg.outdir, 'peb_group', ...
    'peb_hierarchical_up_minus_down_b.mat'), ...
    'PEB3', 'PEBs', 'M3', 'field', '-v7.3');
try
    [BMA3, BMR3] = spm_dcm_peb_bmc(PEB3);
    save(fullfile(cfg.outdir, 'peb_group', ...
        'bma_bmr_hierarchical_up_minus_down_b.mat'), ...
        'BMA3', 'BMR3', 'PEB3', 'PEBs', 'M3', 'field', '-v7.3');
    export_peb_table(BMA3, fullfile(cfg.outdir, 'tables', ...
        'peb_bma_hierarchical_up_minus_down_parameters.tsv'));
catch ME
    warning('Direction hierarchical BMC failed: %s', ME.message);
    export_peb_table(PEB3, fullfile(cfg.outdir, 'tables', ...
        'peb_hierarchical_up_minus_down_parameters.tsv'));
end
end

function [M, design] = build_peb_design(subjects, beh, covariates)
n = numel(subjects);
design = table(subjects(:), 'VariableNames', {'subject'});
X = ones(n,1);
Xnames = {'Intercept'};
for ci = 1:numel(covariates)
    cov = covariates{ci};
    if ~ismember(cov, beh.Properties.VariableNames)
        warning('Skipping absent PEB covariate: %s', cov);
        continue;
    end
    values = nan(n,1);
    for i = 1:n
        idx = find(strcmp(beh.subject, subjects{i}), 1);
        if ~isempty(idx)
            values(i) = double(beh.(cov)(idx));
        end
    end
    if sum(isfinite(values)) < 5 || nanstd_local(values) < 1e-12
        warning('Skipping unusable PEB covariate: %s', cov);
        continue;
    end
    values = zscore_nan(values);
    values(~isfinite(values)) = 0;
    X(:,end+1) = values; %#ok<AGROW>
    Xnames{end+1} = cov; %#ok<AGROW>
    design.(cov) = values;
end
if rank(X) < size(X,2)
    error('Group-level PEB design is rank deficient. Columns: %s', ...
        strjoin(Xnames, ', '));
end
M = struct();
M.X = X;
M.Xnames = Xnames;
M.Q = 'all';
end

function export_peb_table(P, path)
Ep = P.Ep(:);
n = numel(Ep);
if isfield(P, 'Pnames') && numel(P.Pnames) == n
    names = string(P.Pnames(:));
else
    names = "parameter_" + string((1:n)');
end
sd = nan(n,1);
if isfield(P, 'Cp') && all(size(P.Cp) == [n n])
    sd = sqrt(max(diag(P.Cp), 0));
end
pp = nan(n,1);
for i = 1:n
    if isfinite(sd(i)) && sd(i) > 0
        pp(i) = spm_Ncdf(abs(Ep(i))/sd(i));
    end
end
T = table(names, Ep, sd, Ep-1.644854*sd, Ep+1.644854*sd, pp, ...
    'VariableNames', {'parameter','posterior_mean','posterior_sd','ci90_low','ci90_high','posterior_probability_abs_gt_zero'});
writetable(T, path, 'FileType', 'text', 'Delimiter', '\t');
end

function export_run_level_B(records, cfg, path)
rows = {};
for i = 1:numel(records)
    loaded = load(records(i).path, 'DCM');
    D = loaded.DCM;
    names = D.Y.name;
    if ~isfield(D, 'Ep') || ~isfield(D.Ep, 'B')
        continue;
    end
    B = D.Ep.B;
    for src = 1:numel(names)
        for tgt = 1:numel(names)
            if D.b(tgt,src,1) ~= 0
                rows(end+1,:) = {records(i).subject, records(i).run, records(i).condition, ...
                    names{src}, names{tgt}, B(tgt,src,1), cfg.model_variant}; %#ok<AGROW>
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'subject','run','condition','source','target','b_posterior_mean','model_variant'});
writetable(T, path, 'FileType', 'text', 'Delimiter', '\t');
end

% =========================================================================
% Generic helpers
% =========================================================================
function out = to_cellstr_column(values)
if iscell(values)
    out = cellfun(@char, values, 'UniformOutput', false);
elseif isstring(values) || iscategorical(values)
    out = cellstr(string(values));
elseif ischar(values)
    out = cellstr(values);
else
    out = cellstr(string(values));
end
out = out(:);
end

function value = table_text(T, column, row_index)
raw = T.(column);
if iscell(raw)
    value = char(raw{row_index});
elseif isstring(raw) || iscategorical(raw)
    value = char(string(raw(row_index)));
else
    value = char(string(raw(row_index)));
end
end

function T = normalize_subject_column(T)
vars = T.Properties.VariableNames;
col = find_column(vars, {'subject','participant_id','sub'});
if isempty(col)
    error('No subject column found.');
end
subjects = string(T.(col));
for i = 1:numel(subjects)
    token = regexp(char(subjects(i)), '\d+', 'match', 'once');
    if ~isempty(token)
        subjects(i) = sprintf('sub-%02d', str2double(token));
    end
end
T.subject = cellstr(subjects);
end

function col = find_column(vars, candidates)
col = '';
for i = 1:numel(candidates)
    hit = find(strcmpi(vars, candidates{i}), 1);
    if ~isempty(hit)
        col = vars{hit};
        return;
    end
end
end

function path = infer_existing_path(row, candidates, project_root, label)
path = infer_optional_path(row, candidates, project_root);
if isempty(path)
    error('Could not infer existing %s path.', label);
end
end

function path = infer_optional_path(row, candidates, project_root)
path = '';
for i = 1:numel(candidates)
    col = candidates{i};
    if ismember(col, row.Properties.VariableNames)
        value = table_value(row, col);
        if ~isempty(value)
            candidate = resolve_project_path(char(string(value)), project_root);
            if exist(candidate, 'file')
                path = candidate;
                return;
            end
        end
    end
end
end

function value = table_value(row, col)
value = row.(col);
if iscell(value)
    value = value{1};
elseif isstring(value) || iscategorical(value)
    value = value(1);
elseif isnumeric(value) || islogical(value)
    value = value(1);
end
end

function path = resolve_project_path(path, project_root)
path = strrep(path, '<PROJECT_ROOT>', project_root);
path = strrep(path, '${DS000140_ROOT}', project_root);
end

function out = ensure_unzipped_nii(path, cache_dir)
if endsWith(path, '.nii.gz')
    ensure_dir(cache_dir);
    [~, name_gz, ~] = fileparts(path);  % name_gz ends in .nii
    [~, base_name, ~] = fileparts(name_gz);
    out = fullfile(cache_dir, [base_name '.nii']);
    if ~exist(out, 'file')
        files = gunzip(path, cache_dir);
        out = files{1};
    end
else
    out = path;
end
end

function TR = get_TR(row, V)
TR = NaN;
for col = {'tr','TR','repetition_time'}
    if ismember(col{1}, row.Properties.VariableNames)
        value = double(table_value(row, col{1}));
        if isfinite(value) && value > 0
            TR = value;
            return;
        end
    end
end
if isfield(V(1), 'private') && isfield(V(1).private, 'timing')
    TR = double(V(1).private.timing.tspace);
end
if ~isfinite(TR) || TR <= 0
    error('Could not determine TR.');
end
end

function x = standardize_column(x)
x = double(x(:));
x(~isfinite(x)) = nanmean_local(x);
x = x - mean(x);
s = std(x);
if s > 1e-12
    x = x / s;
end
end

function m = nanmean_local(x)
x = x(isfinite(x));
if isempty(x), m = 0; else, m = mean(x); end
end

function s = nanstd_local(x)
x = x(isfinite(x));
if numel(x) < 2, s = 0; else, s = std(x); end
end

function z = zscore_nan(x)
m = nanmean_local(x);
s = nanstd_local(x);
if s <= 1e-12
    z = zeros(size(x));
else
    z = (x - m) / s;
end
end

function b = bool_column(x)
if islogical(x), b = x(:); return; end
s = lower(strtrim(string(x)));
b = s == "true" | s == "1" | s == "yes" | s == "y" | s == "t";
end

function runs = numeric_run_vector(x)
if isnumeric(x)
    runs = double(x(:));
else
    s = string(x(:));
    runs = nan(numel(s),1);
    for i = 1:numel(s)
        token = regexp(char(s(i)), '\d+', 'match', 'once');
        if ~isempty(token), runs(i) = str2double(token); end
    end
end
end

function name = sanitize_name(name)
name = regexprep(name, '[^A-Za-z0-9_]+', '_');
name = regexprep(name, '_+', '_');
name = regexprep(name, '^_|_$', '');
end
