% Evolution to convergence of an observable for an initial wavepacket in a 
% 2D potential with dissipation.

% To collect a lot of data, it is necessary to run only to some level of 
% convergence. See convergenceSchEqLeapfrog.m and the checkConvergence.m
% utility for the structure and parameterization of the convergence 
% criteria.

% Be sure to change DATA_DIR to write to a directory if you so desire.

% Include utilities dir.
addpath(fullfile(pwd, "../../util"));

% Filesystem setup! Important!
WRITE_TO_FS = true;

% Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/NonCVX-2d"; 
benchmark_name = "QP-75d-5s";

%% 2D Setup
dim = 2;
num_cells = 256;

experimentSetup;

%%
% To run a single experiment, use a one element array:
experiments = [...
    ackley_experiment, ...
    ackley2_experiment, ...
    alpine1_experiment, ...
    alpine2_experiment ...
    bohachevsky2_experiment, ...
];

experiments = [...
    ackley_experiment, ...
    ackley2_experiment, ...
    ... %alpine1_experiment, ...
    alpine2_experiment, ...
    bohachevsky2_experiment, ...
];

experiments = [...
    camel3_experiment, ...
    csendes_experiment, ...
    defl_corr_spring_experiment, ...
    dropwave_experiment, ...
    easom_experiment, ...
    griewank_experiment, ...
];

experiments = [...
    holder_experiment, ...
    hosaki_experiment, ...
    levy_experiment, ...
    levy13_experiment, ...
    michalewicz_experiment, ...
    rastrigin_experiment, ...
];

experiments = [...
    rosenbrock_experiment, ...
    shubert_experiment, ...
    styblinski_tang_experiment, ...
    sumofsquares_experiment ...
];

experiments = [...
    ackley_experiment, ...
    ackley2_experiment, ...
    alpine1_experiment, ...
    alpine2_experiment ...
    bohachevsky2_experiment, ...
    camel3_experiment, ...
    csendes_experiment, ...
    defl_corr_spring_experiment, ...
    dropwave_experiment, ...
    easom_experiment, ...
    griewank_experiment, ...
    holder_experiment, ...
    hosaki_experiment, ...
    levy_experiment, ...
    levy13_experiment, ...
    michalewicz_experiment, ...
    rastrigin_experiment, ...
    rosenbrock_experiment, ...
    shubert_experiment, ...
    styblinski_tang_experiment, ...
    sumofsquares_experiment, ...
    xinsheyang3_experiment ...
];

experiments = [defl_corr_spring_experiment, xinsheyang3_experiment];

%%
NUM_STARTS = 1000;

rng("default");
start_points = rand(2, NUM_STARTS);


for tid = 1:numel(experiments)
    exit_statuses = zeros(1, NUM_STARTS);
    run_times = zeros(1, NUM_STARTS);

    % start_points are the same across all functions
    end_points = zeros(2, NUM_STARTS);

    start_values = zeros(1, NUM_STARTS);
    end_values = zeros(1, NUM_STARTS);
    
    experiment = experiments(tid);
    experiment_name = experiment.experiment_dir;
    experiment_V_str = experiment.experiment_V_str;
    
    % Prep directory
    if WRITE_TO_FS
        % Comment at the end of the following line suppresses a warning
        % about potentially unreachable code (intentional switch on
        % WRITE_TO_FS).
        save_target_dir = fullfile(DATA_DIR, benchmark_name); %#ok<*UNRCH>

        if ~exist(save_target_dir, "dir")
            mkdir(save_target_dir);
        end
        % Silence figures if writing them to the filesystem
        fig = figure(tid); set(fig, 'visible','off');
    end
        
    % lower and upper bounds
    lb = [0, 0];
    ub = [1, 1];
    
    
    for start_idx = 1:NUM_STARTS
        x0 = start_points(:, start_idx);
        start_values(start_idx) = experiment.eval_fn(x0(1), x0(2));

        tic
        [x, fval] = fmincon(@(in) experiment.eval_fn(in(1), in(2)), ...
                        x0, ...
                        [], [], ... % A, b
                        [], [], ... % Aeq, beq
                        lb, ub, ...
                        [], optimoptions('fmincon','Algorithm','sqp'));
        run_times(start_idx) = toc;
        
        end_points(:, start_idx) = x;
        end_values(start_idx) = fval;
    end
    
    save_target_fname = strcat(experiment_name, "_fminconData.mat");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, ...
        "start_points", ...
        "end_points", ...
        "start_values", ...
        "end_values", ...
        "run_times" ...
    );
end

%%

figure();
histogram(end_values);

