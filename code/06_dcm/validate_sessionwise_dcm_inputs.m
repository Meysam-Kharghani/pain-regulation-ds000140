function selection = validate_sessionwise_dcm_inputs(variant)
%VALIDATE_SESSIONWISE_DCM_INPUTS Validate one DCM variant and its required inputs.
%
% Checks the exact variant, five ROI definitions, eight unique directed A/B
% edges, ROI-regex uniqueness, mask existence, and required input tables.

variant = char(string(variant));
cfg = get_sessionwise_dcm_config(variant);

required_files = {cfg.inventory_tsv, cfg.roi_definitions_tsv, ...
    cfg.behavior_subjects_tsv};
for i = 1:numel(required_files)
    if ~isfile(required_files{i})
        error('Required input file is missing: %s', required_files{i});
    end
end

if numel(cfg.roi_specs) ~= 5
    error('Variant %s has %d ROIs; expected 5.', ...
        variant, numel(cfg.roi_specs));
end
roi_names = string({cfg.roi_specs.name});
if numel(unique(roi_names)) ~= 5
    error('Variant %s contains duplicate ROI names.', variant);
end

if size(cfg.A_edges, 2) ~= 2
    error('A_edges must have two columns: source and target.');
end
A_source = string(cfg.A_edges(:,1));
A_target = string(cfg.A_edges(:,2));
A_keys = A_source + " -> " + A_target;
if numel(A_keys) ~= 8 || numel(unique(A_keys)) ~= 8
    error('Variant %s must have exactly 8 unique directed A edges.', variant);
end
if any(~ismember(A_source, roi_names)) || any(~ismember(A_target, roi_names))
    error('At least one A edge refers to a node outside roi_specs.');
end

if size(cfg.B_edges,2) ~= 3
    error('B_edges must have three columns: input, source and target.');
end
B_source = string(cfg.B_edges(:,2));
B_target = string(cfg.B_edges(:,3));
B_keys = unique(B_source + " -> " + B_target, 'stable');
if numel(B_keys) ~= 8 || ~isequal(sort(B_keys), sort(A_keys))
    error('The unique B-modulated edges do not match the eight A edges.');
end

R = readtable(cfg.roi_definitions_tsv, ...
    'FileType', 'text', 'Delimiter', '\t');
vars = R.Properties.VariableNames;
name_col = find_column(vars, {'roi','roi_name','name','label'});
mask_col = find_column(vars, {'mask_path','mask','path','file'});
if isempty(name_col) || isempty(mask_col)
    error('Could not identify ROI-name and mask-path columns.');
end

source_roi = strings(5,1);
mask_path = strings(5,1);
pattern = strings(5,1);
for i = 1:5
    pattern(i) = string(cfg.roi_specs(i).pattern);
    names = string(R.(name_col));
    hit = find(~cellfun(@isempty, regexp(cellstr(names), ...
        cfg.roi_specs(i).pattern, 'once')));
    if numel(hit) ~= 1
        error('ROI pattern %s matched %d rows; expected exactly 1.', ...
            cfg.roi_specs(i).pattern, numel(hit));
    end
    source_roi(i) = names(hit);
    mask_path(i) = string(resolve_project_path( ...
        char(string(R.(mask_col)(hit))), cfg.project_root));
    if ~isfile(mask_path(i))
        error('ROI mask does not exist: %s', mask_path(i));
    end
end

selection = table(roi_names(:), pattern, source_roi, mask_path, ...
    'VariableNames', {'dcm_name','regex_pattern','matched_roi','mask_path'});
disp(selection);

fprintf('Validated: 5 ROIs, 8 directed connections, 1 heat input.\n');
fprintf('Expected full analysis: 288 DCMs and 2304 run-level B estimates.\n');
fprintf('VARIANT CONFIG VALIDATION PASSED: %s\n', variant);
end

function col = find_column(vars, candidates)
col = '';
for i = 1:numel(candidates)
    idx = find(strcmpi(vars, candidates{i}), 1);
    if ~isempty(idx)
        col = vars{idx};
        return;
    end
end
end

function path = resolve_project_path(value, project_root)
value = strtrim(value);
if isempty(value)
    path = '';
    return;
end
if is_absolute_path(value)
    path = value;
else
    path = fullfile(project_root, value);
end
end

function tf = is_absolute_path(path)
if ispc
    tf = ~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once')) || ...
        startsWith(path, '\\');
else
    tf = startsWith(path, '/');
end
end
