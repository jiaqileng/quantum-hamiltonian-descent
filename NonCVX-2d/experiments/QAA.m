% Be sure to change DATA_DIR to write to a directory if you so desire.
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/NonCVX-2d";

% Include utilities dir.
addpath(fullfile(pwd, "../util"));

% Include integrator dir
addpath(fullfile(pwd, "../integrators/scheqleapfrog/matlab"));


% Filesystem setup! Important!
WRITE_TO_FS = 1;

%% 2D Setup
dim = 2;
num_cells = 128; % resolution

experimentSetup;

%%
% To run a single experiment, use a one element array:
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


%%
% Switch the for/parfor statement to change whether to run in a parallel
% pool.

% for tid = 1:numel(experiments)
parfor tid = 1:numel(experiments)
    experiment = experiments(tid);
    experiment_dir = experiment.experiment_dir;
    experiment_V_str = experiment.experiment_V_str;
    L = experiment.L;
    X1 = experiment.X1;
    X2 = experiment.X2;
    V = experiment.V;
    H_T = experiment.H_T;
    H_U = experiment.H_U;
    step_size = experiment.step_size;
    
    % Prep directory
    if WRITE_TO_FS
        % Comment at the end of the following line suppresses a warning
        % about potentially unreachable code (intentional switch on
        % WRITE_TO_FS).
        save_target_dir = fullfile(DATA_DIR, experiment_dir); %#ok<*UNRCH>

        if ~exist(save_target_dir, "dir")
            mkdir(save_target_dir);
        end
        % Silence figures if writing them to the filesystem
        fig = figure(tid); set(fig, 'visible','off');
    end
        
    
    % dt should be at most step_size^3 for stability in 2D.
    dt = step_size^(dim+1);
    % approximately 3.33e-6

    % Frame capture rate
    cap_frame_every = 1e5;

    t = 0;
    T = 10;
    
    H_0 = adiabatic_starting_hamiltonian(2 * log2(num_cells));


    % Time dependences on terms in the Hamiltonian
    tdep1 = @(t) 1 - (t / T);
    tdep2 = @(t) t / T;
    
    Re = ones(num_cells^dim, 1) / sqrt(num_cells^dim);
    Im = zeros(num_cells^dim, 1);

    [snapshot_times, wfn] = scheqleapfrog2(Re, Im, ...
                                           H_0, experiment.H_U, ...
                                           T, dt, ...
                                           tdep1, tdep2, ...
                                           cap_frame_every);

    wfn_dims = size(wfn);
    num_frames = wfn_dims(1);
    expected_E = zeros(1, num_frames);
    
    for idx = 1:num_frames
        expected_E(idx) = positionDependentExpectation(wfn(idx,:), experiment.H_U);
    end


    % For fixed endpoint wfn data
    if WRITE_TO_FS
        global_min_loc = experiment.global_min;
        global_min_val = experiment.eval_fn(global_min_loc(1), global_min_loc(2));
        
        % Parallel save
        parsaveFixedT(experiment_dir, save_target_dir, snapshot_times, wfn, expected_E, global_min_val, global_min_loc)

    end
end


function parsaveFixedT(experiment_dir, save_target_dir, snapshot_times, wfn, expected_E, global_min_val, global_min_loc)
    save_target_fname = strcat(experiment_dir, "_QAA_rez128_T10.mat");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, ...
        "snapshot_times", ...
        "wfn", ...
        "expected_E", ...
        "global_min_val", ...
        "global_min_loc", ...
        "-v7.3" ...
    );
end


function H = adiabatic_starting_hamiltonian(n_qubits)
    % From chapter 28 of Andrew Childs' lecture notes
    % The negative of the sum of the Pauli X operator acting on the jth
    % qubit, j from 1 up to n.
    
    % H = -Sum_{jj=0}^{n-1} PauliX^{(jj)}
    
    % Note: 0 up to n-1 is easier to reason about when constructing the 
    % Kronecker product of these matrices). Also, j is reserved in Matlab,
    % so we use the sum variable jj.
    
    X = [0 1; 1 0];
    
    % Equivalent to kron(1, kron(X, speye(2^(n_qubits - 1)))), for jj=0
    H = kron(X, speye(2^(n_qubits-1)));

    % With jj = 0 as the generating case, build up the matrix with a loop
    % over jj from 1 up to n-1.
    for jj = 1:n_qubits-1
        Hj = kron(speye(2^jj), kron(X, speye(2^(n_qubits - jj - 1))));
        H = H + Hj;
    end
    
    H = -sparse(H);
end
