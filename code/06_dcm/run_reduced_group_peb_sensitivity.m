function run_reduced_group_peb_sensitivity(variant)
%RUN_REDUCED_GROUP_PEB_SENSITIVITY Refit group-level PEBs with
% prespecified reduced, low-collinearity designs.
%
% No run-specific DCM is re-estimated. Existing participant-level PEBs are
% loaded from the session-wise all-passive analysis, and a prespecified set
% of reduced third-level designs is fitted:
%
%   all_intercept_only
%   all_up_success
%   all_down_success
%   all_baseline_context
%   direction_intercept_only
%   direction_mean_success
%   direction_success_asymmetry
%   direction_baseline_context
%
% Usage:
%   run_reduced_group_peb_sensitivity('control_value_descending_primary')
%   run_reduced_group_peb_sensitivity('control_value_bilateral_sensitivity')
%
% Requires DS000140_ROOT and SPM25 on the MATLAB path.

if nargin < 1 || isempty(variant)
    variant = 'control_value_bilateral_sensitivity';
end
variant = char(string(variant));

root = getenv('DS000140_ROOT');
if isempty(root)
    error('DS000140_ROOT is not defined.');
end

[spm_name, spm_release] = spm('Ver');
if ~contains(string(spm_name), 'SPM25', 'IgnoreCase', true)
    error('SPM25 is required; found %s %s.', spm_name, spm_release);
end

analysis_name = 'sessionwise_rating_adjusted';
outdir = fullfile(root, 'derivatives', ['dcm_' variant '_' analysis_name]);
subject_peb_dir = fullfile(outdir, 'peb_subjects');
source_table_dir = fullfile(outdir, 'tables');
result_dir = fullfile(outdir, 'peb_reduced_designs');
result_table_dir = fullfile(source_table_dir, 'reduced_peb');

must_be_folder(subject_peb_dir);
must_be_folder(source_table_dir);
ensure_dir(result_dir);
ensure_dir(result_table_dir);

design_file = fullfile(source_table_dir, 'peb_design_all_conditions.tsv');
roi_file = fullfile(source_table_dir, 'dcm_roi_selection.tsv');
if ~isfile(design_file)
    error('Missing design table: %s', design_file);
end
if ~isfile(roi_file)
    error('Missing ROI table: %s', roi_file);
end

D = readtable(design_file, 'FileType', 'text', 'Delimiter', '\t');
D = normalize_subject_column(D);
required_covariates = { ...
    'reg_success_z_up', ...
    'reg_success_z_down', ...
    'passive_mean_rating_z', ...
    'passive_sensory_slope_z', ...
    'up_first'};
for i = 1:numel(required_covariates)
    if ~ismember(required_covariates{i}, D.Properties.VariableNames)
        error('Missing covariate %s in %s.', required_covariates{i}, design_file);
    end
end

subjects = string(D.subject(:));
if numel(subjects) ~= 33 || numel(unique(subjects)) ~= 33
    error('Expected 33 unique participants; found %d rows and %d unique.', ...
        numel(subjects), numel(unique(subjects)));
end

PEBs_all = cell(numel(subjects), 1);
PEBs_direction = cell(numel(subjects), 1);
for i = 1:numel(subjects)
    sub = char(subjects(i));

    f_all = fullfile(subject_peb_dir, sprintf('peb_%s_all_conditions.mat', sub));
    f_direction = fullfile(subject_peb_dir, sprintf('peb_%s_up_minus_down.mat', sub));

    PEBs_all{i} = load_subject_peb(f_all);
    PEBs_direction{i} = load_subject_peb(f_direction);
end

% Use the already group-standardized columns exported by the original PEB.
up_success = zscore_safe(double(D.reg_success_z_up));
down_success = zscore_safe(double(D.reg_success_z_down));
passive_mean = zscore_safe(double(D.passive_mean_rating_z));
passive_slope = zscore_safe(double(D.passive_sensory_slope_z));
order = zscore_safe(double(D.up_first));

mean_success = zscore_safe((up_success + down_success) ./ 2);
success_asymmetry = zscore_safe(up_success - down_success);

specs = {};
specs{end+1} = make_spec('all_intercept_only', 'all', ...
    ones(numel(subjects),1), {'Intercept'});
specs{end+1} = make_spec('all_up_success', 'all', ...
    [ones(numel(subjects),1), up_success, order], ...
    {'Intercept','reg_success_z_up','up_first'});
specs{end+1} = make_spec('all_down_success', 'all', ...
    [ones(numel(subjects),1), down_success, order], ...
    {'Intercept','reg_success_z_down','up_first'});
specs{end+1} = make_spec('all_baseline_context', 'all', ...
    [ones(numel(subjects),1), passive_mean, passive_slope, order], ...
    {'Intercept','passive_mean_rating_z','passive_sensory_slope_z','up_first'});

specs{end+1} = make_spec('direction_intercept_only', 'direction', ...
    ones(numel(subjects),1), {'Intercept'});
specs{end+1} = make_spec('direction_mean_success', 'direction', ...
    [ones(numel(subjects),1), mean_success, order], ...
    {'Intercept','mean_reg_success_z','up_first'});
specs{end+1} = make_spec('direction_success_asymmetry', 'direction', ...
    [ones(numel(subjects),1), success_asymmetry, order], ...
    {'Intercept','success_asymmetry_z_up_minus_down','up_first'});
specs{end+1} = make_spec('direction_baseline_context', 'direction', ...
    [ones(numel(subjects),1), passive_mean, passive_slope, order], ...
    {'Intercept','passive_mean_rating_z','passive_sensory_slope_z','up_first'});

roi_table = readtable(roi_file, 'FileType', 'text', 'Delimiter', '\t');
if ~ismember('dcm_name', roi_table.Properties.VariableNames)
    error('dcm_roi_selection.tsv lacks dcm_name.');
end
roi_names = string(roi_table.dcm_name(:));

all_diagnostics = table();
all_retained = table();

fprintf('Variant: %s\n', variant);
fprintf('Participants: %d\n', numel(subjects));
fprintf('Reduced PEB models: %d\n\n', numel(specs));

for si = 1:numel(specs)
    spec = specs{si};
    fprintf('--- %s ---\n', spec.name);

    if rank(spec.X) < size(spec.X,2)
        error('Design %s is rank deficient.', spec.name);
    end

    if strcmp(spec.kind, 'all')
        PEBs = PEBs_all;
    else
        PEBs = PEBs_direction;
    end

    M3 = struct();
    M3.X = spec.X;
    M3.Xnames = spec.Xnames;
    M3.Q = 'all';
    field = {'B'};

    design_table = array2table(spec.X, 'VariableNames', matlab.lang.makeValidName(spec.Xnames));
    design_table = addvars(design_table, cellstr(subjects), 'Before', 1, ...
        'NewVariableNames', 'subject');
    writetable(design_table, fullfile(result_table_dir, ...
        sprintf('%s_design.tsv', spec.name)), ...
        'FileType', 'text', 'Delimiter', '\t');

    diagnostics = design_diagnostics(spec.name, spec.X, spec.Xnames);
    if isempty(all_diagnostics)
        all_diagnostics = diagnostics;
    else
        all_diagnostics = [all_diagnostics; diagnostics]; %#ok<AGROW>
    end

    PEB3 = spm_dcm_peb(PEBs, M3, field);
    [BMA3, BMR3] = spm_dcm_peb_bmc(PEB3);

    matfile = fullfile(result_dir, sprintf('%s_bma_bmr.mat', spec.name));
    save(matfile, 'BMA3', 'BMR3', 'PEB3', 'PEBs', 'M3', 'field', ...
        'variant', 'analysis_name', '-v7.3');

    outfile = fullfile(result_table_dir, ...
        sprintf('%s_exact_labels.tsv', spec.name));
    T = export_exact_table(BMA3, PEB3, M3, roi_names, outfile);

    keep = T.ci90_excludes_zero;
    if any(isfinite(T.bma_parameter_inclusion_probability))
        keep = keep | T.bma_parameter_inclusion_probability >= 0.95;
    end
    R = T(keep,:);
    if ~isempty(R)
        R = addvars(R, repmat(string(spec.name), height(R), 1), ...
            'Before', 1, 'NewVariableNames', 'reduced_model');
        if isempty(all_retained)
            all_retained = R;
        else
            all_retained = [all_retained; R]; %#ok<AGROW>
        end
    end

    fprintf('Completed %s: %d parameters; %d CI/Pp-retained rows.\n\n', ...
        spec.name, height(T), height(R));
end

writetable(all_diagnostics, fullfile(result_table_dir, ...
    'reduced_design_diagnostics.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

if isempty(all_retained)
    all_retained = table(string.empty(0,1), ...
        'VariableNames', {'reduced_model'});
end
writetable(all_retained, fullfile(result_table_dir, ...
    'reduced_peb_retained_parameters.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

fprintf('\nREDUCED GROUP-PEB SENSITIVITY PASSED\n');
fprintf('Results: %s\n', result_table_dir);
end

function spec = make_spec(name, kind, X, Xnames)
spec = struct();
spec.name = char(name);
spec.kind = char(kind);
spec.X = double(X);
spec.Xnames = Xnames;
end

function P = load_subject_peb(path)
if ~isfile(path)
    error('Missing participant-level PEB: %s', path);
end
S = load(path, 'PEB2');
if ~isfield(S, 'PEB2')
    error('%s does not contain PEB2.', path);
end
P = S.PEB2;
end

function T = export_exact_table(BMA, PEB, M, roi_names, outfile)
if ~isfield(PEB, 'Pnames') || isempty(PEB.Pnames)
    error('PEB.Pnames is absent.');
end
lower_names = string(PEB.Pnames(:));

if isfield(PEB, 'Xnames') && ~isempty(PEB.Xnames)
    group_names = string(PEB.Xnames(:));
elseif isfield(M, 'Xnames') && ~isempty(M.Xnames)
    group_names = string(M.Xnames(:));
else
    error('No group-effect names available.');
end

n_lower = numel(lower_names);
n_group = numel(group_names);
expected_n = n_lower * n_group;

if ~isequal(size(PEB.Ep), [n_lower n_group])
    error('PEB.Ep is %d x %d; labels imply %d x %d.', ...
        size(PEB.Ep,1), size(PEB.Ep,2), n_lower, n_group);
end

mu = double(BMA.Ep(:));
if numel(mu) ~= expected_n
    error('BMA.Ep has %d values; expected %d.', numel(mu), expected_n);
end

parameter_index = (1:expected_n)';
group_effect = repelem(group_names, n_lower, 1);
lower_level_parameter = repmat(lower_names, n_group, 1);

within_subject_effect = strings(expected_n,1);
base_parameter = strings(expected_n,1);
source = strings(expected_n,1);
target = strings(expected_n,1);
connection = strings(expected_n,1);

for i = 1:expected_n
    [within_subject_effect(i), base_parameter(i)] = ...
        split_lower_level_name(lower_level_parameter(i));
    [source(i), target(i), connection(i)] = ...
        map_b_parameter(base_parameter(i), roi_names);
end

parameter = group_effect + " :: " + lower_level_parameter;

sd = nan(expected_n,1);
if isfield(BMA, 'Cp') && isequal(size(BMA.Cp), [expected_n expected_n])
    sd = sqrt(max(diag(double(BMA.Cp)), 0));
end
ci90_low = mu - 1.64485362695147 .* sd;
ci90_high = mu + 1.64485362695147 .* sd;
ci90_excludes_zero = (ci90_low > 0) | (ci90_high < 0);

marginal_sign_probability = nan(expected_n,1);
ok = isfinite(sd) & sd > 0;
marginal_sign_probability(ok) = spm_Ncdf(abs(mu(ok)) ./ sd(ok));

bma_parameter_inclusion_probability = nan(expected_n,1);
if isfield(BMA, 'Pp') && isnumeric(BMA.Pp)
    pp = double(BMA.Pp(:));
    if numel(pp) == expected_n && all(isfinite(pp)) && ...
            all(pp >= 0) && all(pp <= 1)
        bma_parameter_inclusion_probability = pp;
    end
end

full_model_posterior_mean = double(PEB.Ep(:));
bma_minus_full = mu - full_model_posterior_mean;

T = table(parameter_index, group_effect, within_subject_effect, ...
    base_parameter, source, target, connection, lower_level_parameter, ...
    parameter, mu, sd, ci90_low, ci90_high, ci90_excludes_zero, ...
    marginal_sign_probability, bma_parameter_inclusion_probability, ...
    full_model_posterior_mean, bma_minus_full, ...
    'VariableNames', { ...
    'parameter_index','group_effect','within_subject_effect', ...
    'base_parameter','source','target','connection', ...
    'spm_lower_level_parameter_name','parameter', ...
    'posterior_mean','posterior_sd','ci90_low','ci90_high', ...
    'ci90_excludes_zero','marginal_sign_probability', ...
    'bma_parameter_inclusion_probability', ...
    'full_model_posterior_mean','bma_minus_full'});

writetable(T, outfile, 'FileType', 'text', 'Delimiter', '\t');
end

function [within_name, base_name] = split_lower_level_name(value)
s = char(value);
tok = regexp(s, '^([^:]+):\s*(.+)$', 'tokens', 'once');
if isempty(tok)
    within_name = "";
    base_name = string(s);
else
    within_name = string(tok{1});
    base_name = string(tok{2});
end
end

function [source, target, connection] = map_b_parameter(base_name, roi_names)
source = "";
target = "";
connection = "";
s = char(base_name);
tok = regexp(s, 'B\((\d+)\s*,\s*(\d+)\)', 'tokens', 'once');
if isempty(tok)
    return;
end
target_index = str2double(tok{1});
source_index = str2double(tok{2});
if source_index < 1 || source_index > numel(roi_names) || ...
        target_index < 1 || target_index > numel(roi_names)
    return;
end
source = roi_names(source_index);
target = roi_names(target_index);
connection = source + " -> " + target;
end

function T = design_diagnostics(model_name, X, Xnames)
p = size(X,2);
rows = cell(p,7);
for j = 1:p
    name = string(Xnames{j});
    if strcmpi(name, 'Intercept')
        vif = NaN;
        mean_value = mean(X(:,j));
        sd_value = std(X(:,j),0);
    else
        others = X(:, setdiff(1:p,j));
        if isempty(others)
            vif = 1;
        else
            beta = others \ X(:,j);
            residual = X(:,j) - others * beta;
            sse = sum(residual.^2);
            centered = X(:,j) - mean(X(:,j));
            sst = sum(centered.^2);
            if sst <= eps
                vif = Inf;
            else
                r2 = 1 - sse/sst;
                vif = 1/max(1-r2, eps);
            end
        end
        mean_value = mean(X(:,j));
        sd_value = std(X(:,j),0);
    end
    rows(j,:) = {string(model_name), name, size(X,1), rank(X), ...
        mean_value, sd_value, vif}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', ...
    {'reduced_model','predictor','n_subjects','design_rank', ...
    'predictor_mean','predictor_sd','vif'});
end

function values = zscore_safe(values)
values = double(values(:));
if any(~isfinite(values))
    error('A reduced-design variable contains non-finite values.');
end
s = std(values,0);
if s < 1e-12
    error('A reduced-design variable has near-zero variance.');
end
values = (values - mean(values)) ./ s;
end

function T = normalize_subject_column(T)
if ~ismember('subject', T.Properties.VariableNames)
    error('Table lacks subject column.');
end
subjects = string(T.subject);
for i = 1:numel(subjects)
    token = regexp(char(subjects(i)), '\d+', 'match', 'once');
    if isempty(token)
        error('Could not parse subject identifier: %s', subjects(i));
    end
    subjects(i) = sprintf('sub-%02d', str2double(token));
end
T.subject = cellstr(subjects);
end

function must_be_folder(path)
if ~isfolder(path)
    error('Expected folder not found: %s', path);
end
end

function ensure_dir(path)
if ~isfolder(path)
    mkdir(path);
end
end
