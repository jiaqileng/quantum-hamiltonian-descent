
% Evolution to convergence of an observable for an initial wavepacket in a 
% 2D potential with dissipation.

% To collect a lot of data, it is necessary to run only to some level of 
% convergence. See convergenceSchEqLeapfrog.m and the checkConvergence.m
% utility for the structure and parameterization of the convergence 
% criteria.

% Be sure to change DATA_DIR to write to a directory if you so desire.

% Include utilities dir.
addpath(fullfile(pwd, "../../util"));

% Include integrator dir
addpath(fullfile(pwd, "../../integrators/scheqleapfrog/matlab"));


% Filesystem setup
PLOT = 0;
SAVE_PLOT_TO_FS = 0;
SAVE_DATA_TO_FS = 1;

% Please change the data directory
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/NonCVX-2d"; 

%% 2D Setup
dim = 2;
num_cells = 256;

% Run the experiment setup script to init the experiment objects
experimentSetup;

%%

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
    xinsheyang3_experiment, ...
];


%% Experiment Loops

RUN_QHD =                  0;
RUN_GD =                   0;
RUN_NGD =                  1;
RUN_NESTEROV =             1;
RUN_POLYAK =               0;

RUN_CLASSICAL = RUN_GD || RUN_NGD || RUN_NESTEROV || RUN_POLYAK;

syms x1 x2;


for tid = 1:numel(experiments)
    experiment = experiments(tid);
    experiment_dir = experiment.experiment_dir;
    experiment_V_str = experiment.experiment_V_str;
    
    fprintf("thread %d, working on experiment %s\n", tid, experiment_V_str);
   
    L = experiment.L;
    X1 = experiment.X1;
    X2 = experiment.X2;
    V = experiment.V;
    global_min_val = experiment.global_min_val;
    H_T = experiment.H_T;
    H_U = experiment.H_U;
%     surf(X1, X2, reshape(H_U, num_cells, num_cells));
%     break;
    step_size = experiment.step_size;
    
    % dt must be less than the step_size^3 for stability in 2D
    dt = 0.5 * step_size^(dim+1);

    % Frame capture rate
    cap_frame_every = 10;
    

    % Prep directory
    if SAVE_PLOT_TO_FS || SAVE_DATA_TO_FS
        % Comment at the end of the following line suppresses a warning
        % about potentially unreachable code (intentional switch on
        % WRITE_TO_FS).
        save_target_dir = fullfile(DATA_DIR, experiment_dir); %#ok<*UNRCH>

        if ~exist(save_target_dir, "dir")
            mkdir(save_target_dir);
        end
        % Silence figures if writing
        if PLOT
            fig = figure(tid); set(fig,'visible','off');
        end
    else
        fig = figure(tid); set(fig,'visible','on');
    end
    
    if SAVE_DATA_TO_FS
        save_target_fname = strcat(experiment_dir, "_EXPERIMENT");
        save_target_path = fullfile(save_target_dir, save_target_fname);

        % To save: experiment, QHD_times, QHD_loss
        save(save_target_path, 'experiment');
    end

    % Uniform distribution over cells
    psi0_uniform = ones(num_cells^dim, 1) / sqrt(num_cells^dim);
    init_state_str = "$\psi_0$ Uniform(dom)";
    init_state_fname = "uniform";
    

    % QHD
    if RUN_QHD
        fprintf("thread %d, working on exp %s, starting QHD\n", tid, experiment_V_str);
        gamma = 0.1;
        T = 100;

        Re = psi0_uniform;
        Im = zeros(num_cells^2, 1);


        % Poly time dependence
        tdep = PolyTimeDep(gamma, 2);

        [QHD_times, QHD_wfn]= scheqleapfrog(Re, Im, ...
                                            H_T, H_U, ...
                                            T, dt, @tdep.eval_tdep, ...
                                            cap_frame_every);

        num_frames = length(QHD_times);
        QHD_loss = zeros(1, num_frames);

        for k = 1:num_frames
            wfn = QHD_wfn(k,:);
            prob = abs(wfn).^2;
            QHD_loss(k) = dot(prob, H_U);
        end
        
        % Write QHD data to file system
        if SAVE_DATA_TO_FS
            save_target_fname = strcat(experiment_dir, "_QHD");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            
            % To save: experiment, QHD_times, QHD_loss
            save(save_target_path, 'QHD_times', 'QHD_loss');
        end

        if PLOT
            clf;
            x0 = 10;
            y0 = 10;
            width = 720;
            height = 480;
            set(gcf, 'position', [x0, y0, width, height]);

            plot(QHD_times, QHD_loss);
            yline(global_min_val);

            title_fmt_str = "Expected potential energy of evolution of %s in $V=$ %s with %s";
            title_str = sprintf(title_fmt_str, init_state_str, experiment_V_str, tdep.gen_title());

            title(title_str, "interpreter", "latex");
            xlabel("Time");
            ylabel("Expected Potential Energy");
%             ylim([global_min_val-2 inf]);
        
            if SAVE_PLOT_TO_FS
                save_target_fname = strcat(experiment_dir, "_", init_state_fname, "_", tdep.gen_fname(), ".png");
                save_target_path = fullfile(save_target_dir, save_target_fname);
                saveas(fig, save_target_path);
                pause(1);
            end
        end
        fprintf("thread %d, working on exp %s, finished QHD\n", tid, experiment_V_str);
    end % QHD
    
    
    % Block for classical setup
    if RUN_CLASSICAL
        % Number of starting points
        N = 1000;
        
        MAX_FRAMES = 10000;
        
        rng('default');
        starting_points = rand(2, N);
        
        save('starting_points', 'starting_points');

    end % Classical setup
    
    
    if RUN_GD
        fprintf("thread %d, working on exp %s, starting classical GD\n", tid, experiment_V_str);
        
        gd_fn_vals = zeros(N,1);
        
        for idx = 1:N
            x = starting_points(:, idx);
            grad = experiment.eval_grad(x(1), x(2));

            % Step size, eta
            eta = 1e-3;
            
            % Convergence parameter in gradient, epsilon
            conv_eps = 1e-8;
            
            step_count = 0;
            frame_count = 0;
            
            % Rescales x axis of plot
            cap_frame_every = 1;

            while norm(grad) > conv_eps && frame_count < MAX_FRAMES
                x = x - eta * grad;
                grad = experiment.eval_grad(x(1), x(2));
                step_count = step_count + 1;

                if mod(step_count, cap_frame_every) == 0
                    frame_count = frame_count + 1;
                    res_idx = step_count / cap_frame_every;
                    gd_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                end
            end
            
            if frame_count == 0
                gd_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
                gd_positions(idx, 1, :) = x;
                gd_last_frame(idx) = 1;
            end
        end
            
        if SAVE_DATA_TO_FS
            save_target_fname = strcat(experiment_dir, "_GD");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            save(save_target_path, 'gd_fn_vals');
        end

        if PLOT
            figure(1);

            % Can convert direct plot to jagged array by way of cell array
            line = plot(1:frame_count, gd_fn_vals(idx, 1:frame_count));
            try
                line.Color = [0, 0, 0, 0.5];
            catch e
            end
            hold on;
        end

        if SAVE_PLOT_TO_FS
            % Save/graph
            title_fmt_str = "Gradient descent of %d starting points in $V=$ %s to %g convergence";
            title_str = sprintf(title_fmt_str, N, experiment_V_str, conv_eps);

            title(title_str, "interpreter", "latex");
            xlabel("GD Steps");
            ylabel("Energy");
            
            save_target_fname = strcat(experiment_dir, "_GD.png");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            saveas(fig, save_target_path);
            pause(1);
        end
        fprintf("thread %d, working on exp %s, finished classical GD\n", tid, experiment_V_str);
    end % GD
    
    
    if RUN_NGD
        fprintf("thread %d, working on exp %s, starting classical noisy GD \n", tid, experiment_V_str);
                
        ngd_fn_vals = zeros(N, MAX_FRAMES);
        ngd_positions = zeros(N, MAX_FRAMES, 2);
        ngd_last_frame = zeros(N, 1);

        % Distribution of noise
        mu = 0;
        sigma = 1;
                
        for idx = 1:N
            if mod(idx-1, 100) == 0
                disp(idx-1);
            end
            
            x = starting_points(:, idx);
            grad = experiment.eval_grad(x(1), x(2));

            % Step size            
            alpha = 1e-2;
            
            % Convergence parameter in gradient, epsilon
            conv_eps = 1e-8;
            
            step_count = 0;
            frame_count = 0;
            
            % Rescales x axis of plot
            cap_frame_every = 1;

            while norm(grad) > conv_eps && frame_count < MAX_FRAMES
                eta = alpha/(1 + alpha * step_count);

                x = x - eta * (grad + normrnd(mu, sigma, 2, 1));
                
                % Project back inside [0,1]x[0,1] boundaries
                if x(1) < 0
                    x(1) = 0;
                end
                
                if x(2) < 0
                    x(2) = 0;
                end
                
                if x(1) > 1
                    x(1) = 1;
                end
                
                if x(2) > 1
                    x(2) = 1;
                end
                
                grad = experiment.eval_grad(x(1), x(2));
                
                step_count = step_count + 1;

                if mod(step_count, cap_frame_every) == 0
                    frame_count = frame_count + 1;
                    res_idx = step_count / cap_frame_every;
                    ngd_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                    ngd_positions(idx, res_idx, :) = x;
                    ngd_last_frame(idx) = frame_count;
                end
            end
            
            if frame_count == 0
                ngd_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
                ngd_positions(idx, 1, :) = x;
                ngd_last_frame(idx) = 1;
            end
        end
        
        ngd_fn_vals = ngd_fn_vals(:, 1:max(ngd_last_frame));
        ngd_positions = ngd_positions(:, 1:max(ngd_last_frame), :);
            
        if SAVE_DATA_TO_FS
            save_target_fname = strcat(experiment_dir, "_NGD");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            save(save_target_path, 'ngd_fn_vals', 'ngd_positions', 'ngd_last_frame');
        end

        if PLOT
            figure(1);

            % Can convert direct plot to jagged array by way of cell array
            line = plot(1:frame_count, ngd_fn_vals(idx, 1:frame_count));
            try
                line.Color = [0, 0, 0, 0.5];
            catch e
            end
            hold on;
        end

        if SAVE_PLOT_TO_FS
            % Save/graph
            title_fmt_str = "Gradient descent of %d starting points in $V=$ %s to %g convergence";
            title_str = sprintf(title_fmt_str, N, experiment_V_str, conv_eps);

            title(title_str, "interpreter", "latex");
            xlabel("NGD Steps");
            ylabel("Energy");
            
            save_target_fname = strcat(experiment_dir, "_SGD.png");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            saveas(fig, save_target_path);
            pause(1);
        end
        fprintf("thread %d, working on exp %s, finished classical NGD\n", tid, experiment_V_str);
    end % NGD
    
    
    if RUN_NESTEROV
        % http://www.princeton.edu/~yc5/ele522_optimization/lectures/accelerated_gradient.pdffprintf("thread %d, working on exp %s, starting classical Polyak\n", tid, experiment_V_str);
        fprintf("thread %d, working on exp %s, starting classical Nesterov\n", tid, experiment_V_str);

        nesterov_fn_vals = zeros(N, MAX_FRAMES);
        nesterov_positions = zeros(N, MAX_FRAMES, 2);
        nesterov_last_frame = zeros(N, 1);

        % Nesterov
        % Step size
        eta = 1e-4;

        for idx = 1:N
            if mod(idx-1, 100) == 0
                disp(idx-1);
            end
            
            % initial values (selected randomly earlier)
            x = starting_points(:, idx);
            y = starting_points(:, idx);
            x_last = starting_points(:, idx);
            y_last = starting_points(:, idx);
            
            grad_y = experiment.eval_grad(y(1), y(2));
                        
            % Convergence in gradient, epsilon
            conv_eps = 1e-8;
            
            step_count = 0;
            frame_count = 0;
            
            % Rescales x axis of plot
            cap_frame_every = 1;

            while norm(grad_y) > conv_eps && frame_count < 10000
                x_last = x;
                y_last = y;
                
                x = y - eta * grad_y;
                
                % Project back inside [0,1]x[0,1] boundaries
                if x(1) < 0
                    x(1) = 0;
                end
                
                if x(2) < 0
                    x(2) = 0;
                end
                
                if x(1) > 1
                    x(1) = 1;
                end
                
                if x(2) > 1
                    x(2) = 1;
                end
                
                y = x + (step_count / (step_count + 3)) * (x - x_last);
                
                % Project back inside [0,1]x[0,1] boundaries
                if y(1) < 0
                    y(1) = 0;
                end
                
                if y(2) < 0
                    y(2) = 0;
                end
                
                if y(1) > 1
                    y(1) = 1;
                end
                
                if y(2) > 1
                    y(2) = 1;
                end

                grad_y = experiment.eval_grad(y(1), y(2));
                
                step_count = step_count + 1;

                if mod(step_count, cap_frame_every) == 0
                    frame_count = frame_count + 1;
                    res_idx = step_count / cap_frame_every;
                    nesterov_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                    nesterov_positions(idx, res_idx, :) = y;
                    nesterov_last_frame(idx) = frame_count;
                end
            end
            
            if frame_count == 0
                nesterov_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
                nesterov_positions(idx, 1, :) = x;
                nesterov_last_frame(idx) = 1;
            end
        end
        
        nesterov_fn_vals = nesterov_fn_vals(:, 1:max(nesterov_last_frame));
        nesterov_positions = nesterov_positions(:, 1:max(nesterov_last_frame), :);

        if SAVE_DATA_TO_FS
            save_target_fname = strcat(experiment_dir, "_NESTEROV");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            save(save_target_path, 'nesterov_fn_vals', 'nesterov_positions', 'nesterov_last_frame');
        end

        if PLOT
            figure(3);

            % Can convert direct plot to jagged array by way of cell array
            line = plot(1:frame_count, nesterov_fn_vals(idx, 1:frame_count));
            try
                line.Color = [0, 0, 0, 0.5];
            catch e
            end
            hold on;
        end
        
        fprintf("thread %d, working on exp %s, finished classical Nesterov\n", tid, experiment_V_str);
    end % Nesterov
    
    
    if RUN_POLYAK
        % http://www.princeton.edu/~yc5/ele522_optimization/lectures/accelerated_gradient.pdf
        fprintf("thread %d, working on exp %s, starting classical Polyak\n", tid, experiment_V_str);
        
        polyak_fn_vals = zeros(N,1);
        
        % Polyak hyperparameters
        % Step size
        eta = 1e-4;
        theta = 0.2;

        for idx = 1:N
            x = starting_points(:, idx);
            x_last = x;
            

            % Convergence in gradient, epsilon
            conv_eps = 1e-3;
            
            step_count = 1;
            frame_count = 1;
            
            % Rescales x axis of plot
            cap_frame_every = 10;

            
            % First step along gradient, later steps have momentum
            grad = experiment.eval_grad(x(1), x(2));
            x = x_last - eta * grad;    
            
            while norm(grad) > conv_eps && frame_count < 10000
                x_next = x - eta * grad + theta * (x - x_last);
                
                % Advance assignments
                x_last = x;
                x = x_next;
                
                % Recompute gradients with next step
                grad = experiment.eval_grad(x(1), x(2));
                
                step_count = step_count + 1;

                if mod(step_count, cap_frame_every) == 0
                    frame_count = frame_count + 1;
                    res_idx = step_count / cap_frame_every;
                    polyak_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                end
            end
            
            if frame_count == 0
                polyak_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
                polyak_positions(idx, 1, :) = x;
                polyak_last_frame(idx) = 1;
            end
        end
            
        if SAVE_DATA_TO_FS
            save_target_fname = strcat(experiment_dir, "_POLYAK");
            save_target_path = fullfile(save_target_dir, save_target_fname);
            save(save_target_path, 'polyak_fn_vals');
        end

        if PLOT
            figure(2);

            % Can convert direct plot to jagged array by way of cell array
            line = plot(1:frame_count-1, polyak_fn_vals(idx, 1:frame_count-1));
            try
                line.Color = [0, 0, 0, 0.5];
            catch e
            end
            hold on;
        end
        fprintf("thread %d, working on exp %s, finished classical Polyak \n", tid, experiment_V_str);
    end % Polyak
end % N samples
