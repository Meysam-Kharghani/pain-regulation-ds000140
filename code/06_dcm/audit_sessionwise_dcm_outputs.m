function report = audit_sessionwise_dcm_outputs(variant, scope)
%AUDIT_SESSIONWISE_DCM_OUTPUTS Audit completed outputs for one run-separated DCM variant.
%
% scope: smoke, full, or auto.

if nargin < 2 || isempty(scope)
    scope = 'auto';
end
variant = char(string(variant));
scope = lower(char(string(scope)));

cfg = get_sessionwise_dcm_config(variant);
summary_file = fullfile(cfg.outdir, 'tables', ...
    'sessionwise_dcm_run_summary.tsv');
B_file = fullfile(cfg.outdir, 'tables', ...
    'sessionwise_run_b_parameters.tsv');

if ~isfile(summary_file)
    error('Missing DCM summary: %s', summary_file);
end
if ~isfile(B_file)
    error('Missing run-level B table: %s', B_file);
end

T = readtable(summary_file, 'FileType', 'text', 'Delimiter', '\t');
conditions = lower(strtrim(string(T.condition)));
statuses = upper(strtrim(string(T.status)));
subjects = string(T.subject);

if strcmp(scope, 'auto')
    if height(T) == 9 && numel(unique(subjects)) == 1
        scope = 'smoke';
    elseif height(T) == cfg.expected_n_dcm_runs
        scope = 'full';
    else
        error('Cannot infer audit scope from %d summary rows.', height(T));
    end
end

switch scope
    case 'smoke'
        expected_total = 9;
        expected_subjects = 1;
        expected_passive = 7;
        expected_up = 1;
        expected_down = 1;
    case 'full'
        expected_total = cfg.expected_n_dcm_runs;
        expected_subjects = cfg.expected_total_subjects;
        expected_passive = cfg.expected_n_passive_runs;
        expected_up = cfg.expected_n_up_runs;
        expected_down = cfg.expected_n_down_runs;
    otherwise
        error('Unknown scope: %s', scope);
end

assert(height(T) == expected_total, ...
    'Expected %d summary rows; found %d.', expected_total, height(T));
assert(numel(unique(subjects)) == expected_subjects, ...
    'Expected %d participants; found %d.', ...
    expected_subjects, numel(unique(subjects)));
assert(all(statuses == "OK"), 'At least one run does not have status OK.');
assert(sum(conditions == "passive") == expected_passive, ...
    'Passive-run count mismatch.');
assert(sum(conditions == "up") == expected_up, 'Up-run count mismatch.');
assert(sum(conditions == "down") == expected_down, 'Down-run count mismatch.');
assert(all(double(T.n_regions) == numel(cfg.roi_specs)), ...
    'At least one DCM has an unexpected number of regions.');

passive = conditions == "passive";
regulation = conditions == "up" | conditions == "down";
assert(all(double(T.n_stim_events(passive)) == 11));
assert(all(double(T.n_rating_events(passive)) == 11));
assert(all(double(T.n_stim_events(regulation)) == 10));
assert(all(double(T.n_rating_events(regulation)) == 10));

B = readtable(B_file, 'FileType', 'text', 'Delimiter', '\t');
B_keys = unique(string(cfg.B_edges(:,2)) + " -> " + ...
    string(cfg.B_edges(:,3)), 'stable');
expected_B_per_run = numel(B_keys);
expected_B_rows = expected_total * expected_B_per_run;
assert(height(B) == expected_B_rows, ...
    'Expected %d run-level B rows; found %d.', expected_B_rows, height(B));
assert(ismember('b_posterior_mean', B.Properties.VariableNames), ...
    'B table lacks b_posterior_mean.');
assert(all(isfinite(double(B.b_posterior_mean))), ...
    'Run-level B table contains non-finite values.');

mat_files = dir(fullfile(cfg.outdir, 'estimated_dcms', '*_estimated.mat'));
assert(numel(mat_files) == expected_total, ...
    'Expected %d estimated DCM MAT files; found %d.', ...
    expected_total, numel(mat_files));

problems = strings(0,1);
for i = 1:numel(mat_files)
    filename = fullfile(mat_files(i).folder, mat_files(i).name);
    S = load(filename, 'DCM');
    if ~isfield(S, 'DCM')
        problems(end+1) = "Missing DCM variable: " + filename; %#ok<AGROW>
        continue;
    end
    D = S.DCM;

    if ~isfield(D, 'F') || ~isscalar(D.F) || ~isfinite(D.F)
        problems(end+1) = "Invalid free energy: " + filename; %#ok<AGROW>
    end
    if ~isfield(D, 'Ep')
        problems(end+1) = "Missing posterior means: " + filename; %#ok<AGROW>
    else
        try
            ep = full(spm_vec(D.Ep));
            if isempty(ep) || any(~isfinite(ep))
                problems(end+1) = "Invalid posterior means: " + filename; %#ok<AGROW>
            end
        catch ME
            problems(end+1) = "Could not vectorize Ep: " + filename + ...
                " | " + string(ME.message); %#ok<AGROW>
        end
    end
    if ~isfield(D, 'Cp')
        problems(end+1) = "Missing posterior covariance: " + filename; %#ok<AGROW>
    else
        Cp = full(D.Cp);
        if isempty(Cp) || any(~isfinite(Cp(:))) || ...
                size(Cp,1) ~= size(Cp,2)
            problems(end+1) = "Invalid posterior covariance: " + filename; %#ok<AGROW>
        elseif any(diag(Cp) < -1e-8)
            problems(end+1) = "Negative posterior variances: " + filename; %#ok<AGROW>
        end
    end
    if ~isfield(D, 'n') || D.n ~= numel(cfg.roi_specs)
        problems(end+1) = "Unexpected DCM.n: " + filename; %#ok<AGROW>
    end
    if ~isfield(D, 'b') || nnz(D.b) ~= expected_B_per_run
        problems(end+1) = "Unexpected B-design edge count: " + filename; %#ok<AGROW>
    end
end

if ~isempty(problems)
    fprintf('\nAudit problems:\n');
    disp(problems);
    error('Sessionwise DCM audit failed for %s.', variant);
end

report = table(string(variant), string(scope), expected_subjects, ...
    expected_total, expected_passive, expected_up, expected_down, ...
    expected_B_rows, numel(mat_files), ...
    'VariableNames', {'variant','scope','n_subjects','n_dcms', ...
    'n_passive','n_up','n_down','n_b_rows','n_estimated_mat_files'});
disp(report);
fprintf('SESSIONWISE DCM AUDIT PASSED: %s (%s)\n', variant, scope);
end
