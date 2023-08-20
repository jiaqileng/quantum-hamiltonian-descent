function [snapshot_times, psi] = scheqleapfrog2(Re, Im, ...
                                               H_T, H_U, ...
                                               T, dt, tdep1, tdep2, ...
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
        td1 = tdep1(t);
        td2 = tdep2(t);
        Im_half = Im + (dt/2) .* (td1 .* (H_T * Re) + td2 .* H_U .* Re);
        
        t = t + dt/2;
        td1 = tdep1(t);
        td2 = tdep2(t);
        
        Re = Re - dt .* (td1 .* (H_T * Im_half) + td2 .* H_U .* Im_half);
        Im = Im_half + (dt/2) .* (td1 .* (H_T * Re) + td2 .* H_U .* Re);
        
        t = t + dt/2;

        if mod(step, cap_frame_every) == 0
            frame_idx = 1 + step / cap_frame_every;
            snapshot_times(frame_idx) = t;
            psi(frame_idx,:) = Re + 1i.*Im;
            
            if mod(frame_idx, 100) == 0
                disp(frame_idx);
            end
        end
    end
    
    leapfrog_runtime = toc;
    fprintf('Integrator running time = %d\n', leapfrog_runtime);
end
