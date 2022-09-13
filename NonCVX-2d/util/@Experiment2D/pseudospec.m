function [snapshot_times, psi] = pseudospec(obj, init_type,...
                                            T, tdep1, tdep2,...
                                            cap_frame_every)
    % PSEUDOSPEC provides a pseudo-spectral solver for the time-dependent
    % Schrodinger equation with Hamiltonian 
    %   H(t) = -0.5 * tdep(t) * Laplacian + V(x)
    % defined on the unit square [0,1]^2 with 
    % Dirichlet boundary condition.
    
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
    
    % Step 2: discretization in the time domain
    dt = 0.5 * obj.step_size.^2;
    num_steps = ceil(T/dt);
    approx_num_frames = (num_steps / cap_frame_every);

    if abs(approx_num_frames - round(approx_num_frames)) < 10^-10
        num_frames = round(approx_num_frames);
    else
        num_frames = ceil(num_steps / cap_frame_every);
    end
    
    num_steps = cap_frame_every * num_frames;

    dt = T / num_steps;% So the new dt will always be less than or equal to the initial guess.
    t = 0;
    
    % step 3: to build the kinetic operator in the freq. space
    truncated_bd = floor(obj.cells_along_dim/2);
    k1 = -truncated_bd : truncated_bd - 1;
    k2 = -truncated_bd : truncated_bd - 1;
    [k1, k2] = meshgrid(k1, k2);
    kinetic = 2*pi^2.*(k1.^2 + k2.^2);
   
    % step 4: 2nd-order Trotter in time propagation
    snapshot_times = zeros(num_frames + 1, 1);
    psi = zeros(num_frames + 1, obj.cells_along_dim^2);
    
    psi(1,:) = psi0(:);
    tic
    for step = 1:num_steps
        coeffT = tdep1(t+dt/2);
        coeffU = tdep2(t+dt/2);
        u1 = exp(-1i.*(dt/2).*coeffU.*obj.V).*psi0;
        u2 = ifft2(ifftshift( (exp(-1i.*dt.*coeffT.*kinetic) .* fftshift(fft2(u1))) ));
        psi_new = exp(-1i.*(dt/2).*coeffU.*obj.V).*u2;
        t = t + dt;
        
        if mod(step, cap_frame_every) == 0
            frame_idx = 1 + step / cap_frame_every;
            snapshot_times(frame_idx) = t;
            psi(frame_idx,:) = psi_new(:);
        end
        
        psi0 = psi_new;
    end
    
    pseudospec_runtime = toc;
    fprintf('Pseudospectral integrator running time = %d\n', pseudospec_runtime);
    
end