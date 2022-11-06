% Set resolution to 128 or 256
num_cells = 256
run experimentSetup.m

num_experiments = 22;
names = cell(1, num_experiments);
potentials = zeros(num_experiments, num_cells, num_cells);
for idx = 1:num_experiments
    experiment = experiments(idx);
    names{idx} = convertStringsToChars(experiment.experiment_dir);
    potentials(idx, :, :) = experiment.V(:, :);
end

save(strcat("potentials", int2str(num_cells)), "names", "potentials")