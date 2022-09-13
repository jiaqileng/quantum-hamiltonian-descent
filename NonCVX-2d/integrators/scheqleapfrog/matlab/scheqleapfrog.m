function [snapshot_times, psi] = scheqleapfrog(Re, Im, ...
                                               H_T, H_U, ...
                                               T, dt, tdep, ...
                                               cap_frame_every)
    % Schrodinger equation leapfrog solver

    total_cells = numel(Re);
    num_steps = T / dt;
    
    approx_num_frames = (num_steps / cap_frame_every);

    if abs(approx_num_frames - round(approx_num_frames)) < 10^-10
        num_frames = round(approx_num_frames);
    else
        num_frames = ceil(num_steps / cap_frame_every);
    end
    
    num_steps = cap_frame_every * num_frames;

    % So the new dt will always be less than or equal to the initial guess.
    dt = T / num_steps;
    
    
    snapshot_times = zeros(num_frames + 1, 1);
    psi = zeros(num_frames + 1, total_cells);
    
    psi(1,:) = Re + 1i.*Im;

    t = 0;
    tic
    for step = 1:num_steps
        Im_half = Im + (dt/2) .* ((tdep(t) .* (H_T * Re)) + (H_U .* Re));
        t = t + dt/2;
        Re = Re - dt .* ((tdep(t) .* (H_T * Im_half)) + (H_U .* Im_half));
        Im = Im_half + (dt/2) .* ((tdep(t) .* (H_T * Re)) + (H_U .* Re));
        t = t + dt/2;

        if mod(step, cap_frame_every) == 0
            frame_idx = 1 + step / cap_frame_every;
            snapshot_times(frame_idx) = t;
            psi(frame_idx,:) = Re + 1i.*Im;
        end
    end
    
    leapfrog_runtime = toc;
    fprintf('FDM + leapfrog integrator running time = %d\n', leapfrog_runtime);
end
