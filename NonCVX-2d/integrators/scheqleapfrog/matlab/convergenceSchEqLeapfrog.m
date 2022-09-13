function [snapshot_times, wfn, expected_observable] = convergenceSchEqLeapfrog(Re, Im, ...
                                                           H_T, H_U, ...
                                                           dt, tdep_fn, ...
                                                           observable_fn, lookback_offsets, percent_diff, ...
                                                           cap_frame_every)

    % Schrodinger equation leapfrog solver to convergence of tolerance tol.
    % Modified to include the wavefunction.

    MAX_FRAMES = 1.5e4;
%     MAX_SIM_TIME = 250;
    
    % Redundant if moved to a run method of the experiment class; there
    % would be no cost to pass both H_U and V, or simply not need to store
    % both.
    
    % Compute the (possibly unnormalized) expected potential energy.
    observable_expectation_fn = @(wave) observable_fn(wave, H_U);
    
    % Add 1 to account for initial wave.
    snapshot_times = zeros(1, 1);
    expected_observable = zeros(1, 1);

    snapshot_times(1) = 0;  % Redundant to the init, but explicitly set.
    wfn = zeros(1, numel(Re));
    wfn(1, :) = Re + 1i.*Im;
    expected_observable(1) = observable_expectation_fn((Re + 1i.*Im).');

    % Start simulation at time 0.
    t = 0;
    
    % Number of times simulation has stepped forward by dt
    step = 1;
    
    % Number of frames that have been saved. Start at index 2 because the 
    % initial wave is saved at index 1.
    frame_idx = 2;
    
    % How often to check convergence depends on how far back we're looking
    % with each check.
    furthest_lookback = max(lookback_offsets);
    fraction_to_advance = 0.1;
    
    
    tic
    while (frame_idx <= 1+furthest_lookback) || ((frame_idx < MAX_FRAMES) && ~checkConvergence(expected_observable, lookback_offsets, percent_diff))
        last_frame_idx = frame_idx;
        
        % Advance simulation by about a quarter of the furthest lookback before
        % testing convergence again.
        while frame_idx <= 1+furthest_lookback || frame_idx < last_frame_idx + round(fraction_to_advance * furthest_lookback)
        
            Im_half = Im + (dt/2) .* ((tdep_fn(t) .* (H_T * Re)) + (H_U .* Re));
            t = t + dt/2;
            Re = Re - dt .* ((tdep_fn(t) .* (H_T * Im_half)) + (H_U .* Im_half));
            Im = Im_half + (dt/2) .* ((tdep_fn(t) .* (H_T * Re)) + (H_U .* Re));
            t = t + dt/2;

            if mod(step, cap_frame_every) == 0
                snapshot_times(frame_idx) = t;
                wfn(frame_idx, :) = Re + 1i.*Im;
                expected_observable(frame_idx) = observable_expectation_fn((Re + 1i.*Im).');
                frame_idx = frame_idx + 1;
            end

            step = step + 1;
        end
    end
    leapfrog_runtime = toc;
    
    fprintf('Convergence integrator running time = %d\n', leapfrog_runtime);
    fprintf('Convergence detected at sim time = %d\n', t);
end
