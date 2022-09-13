function [snapshot_times, psi] = fdmleapfrog(obj, ...
                                             init_type,...
                                             T, ...
                                             tdep1, ...
                                             tdep2,...
                                             cap_frame_every)                  
    % Schrodinger equation leapfrog solver
    
    % step 1: to build initial condition
    if isa(init_type, 'string')
        if init_type == "uniform_superposition"
            psi0 = ones(obj.cells_along_dim, obj.cells_along_dim)./obj.cells_along_dim;
        elseif init_type == "sine"
            ground_state = @(x1, x2) sin(pi.*x1).*sin(pi.*x2);
            psi0 = ground_state(obj.X1, obj.X2);
            psi0 = psi0./norm(psi0(:));
        else 
            fprintf("init_type not correct! Input string 'uniform_superposition' or 'sine', or numerical vector data. Program breaks.")
            return;
        end
    elseif (isa(init_type, "double") || isa(init_type, "complex"))
        [dim1, dim2] = size(init_type);
        if (dim1 == obj.cells_along_dim && dim2 == obj.cells_along_dim)
            psi0 = init_type;
        elseif (dim1 == obj.cells_along_dim^2 && dim2 == 1)
            psi0 = reshape(init_type, obj.cells_along_dim, obj.cells_along_dim);
        elseif (dim1 == 1 && dim2 == obj.cells_along_dim^2)
            psi0 = reshape(init_type, obj.cells_along_dim, obj.cells_along_dim);
        else
            fprintf("init_type not correct! Input string 'uniform_superposition' or 'sine', or numerical vector data. Program breaks.")
            return;
        end
    else
        fprintf("init_type not correct! Input string 'uniform_superposition' or 'sine', or numerical vector data. Program breaks.")
            return;
    end
    
    Re = real(psi0(:));
    Im = imag(psi0(:));
    dt = obj.step_size^3;
    total_cells = numel(Re);
    num_steps = T / dt;
    
    approx_num_frames = (num_steps / cap_frame_every);

    if abs(approx_num_frames - round(approx_num_frames)) < 10^-10
        num_frames = round(approx_num_frames);
    else
        num_frames = ceil(num_steps / cap_frame_every);
    end
    
    disp("total frames: " + string(num_frames+1));
    
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
        Im_half = Im + (dt/2) .* (td1 .* (obj.H_T * Re) + td2 .* obj.H_U .* Re);
        
        t = t + dt/2;
        td1 = tdep1(t);
        td2 = tdep2(t);
        
        Re = Re - dt .* (td1 .* (obj.H_T * Im_half) + td2 .* obj.H_U .* Im_half);
        Im = Im_half + (dt/2) .* (td1 .* (obj.H_T * Re) + td2 .* obj.H_U .* Re);
        
        t = t + dt/2;

        if mod(step, cap_frame_every) == 0
            frame_idx = 1 + step / cap_frame_every;
            snapshot_times(frame_idx) = t;
            psi(frame_idx,:) = Re + 1i.*Im;
            if mod(frame_idx, 500) == 0
                disp(frame_idx);
            end
        end
    end
    
    leapfrog_runtime = toc;
    fprintf('FDM + leapfrog integrator running time = %d\n', leapfrog_runtime);
end