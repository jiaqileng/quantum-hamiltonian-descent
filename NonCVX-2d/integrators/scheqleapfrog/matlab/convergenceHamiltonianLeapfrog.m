function [snapshot_times, position, velocity, observable] = convergenceHamiltonianLeapfrog( ...
                                                               x_0, ...
                                                               v_0, ...
                                                               fn_oracle, ...
                                                               gradient_fn, ...
                                                               bounds, ...
                                                               dt, ...
                                                               tdep_fn, ...
                                                               min_frames, ...
                                                               max_frames, ...
                                                               v_norm_cutoff, ...
                                                               cap_frame_every ...
                                                           )
%CONVERGENCEHAMILTONIANLEAPFROG Classical Hamiltonian dynamics with time
%dependent update decay and an update velocity cutoff for convergence
%detection.


t = 0;
x = x_0;
v = v_0;
a = -1 .* gradient_fn(x_0(1), x_0(2))';

snapshot_times = zeros(1, 1);
position = zeros(1, 2);
velocity = zeros(1, 2);
observable = zeros(1);

step = 0;
frame_idx = 1;


% First run at least as many frames as you want to 'look back' on for
% convergence. Otherwise run to the max number of frames you want, and
% check whether the velocity is going to update the position by "enough" to
% make it worthwile to continue the loop. Otherwise declare convergence and
% halt.
while (frame_idx < min_frames) || (((tdep_fn(t) * norm(v)) > v_norm_cutoff) && (frame_idx < max_frames))
    v_half = v + (dt/2) .* a;
    t = t + dt/2;

    % Implements reflection off of the boundaries. If a point
    % would land outside of the domain after being updated by v_half,
    % flip the velocity component along the dimension being violated.
    % In rare cases it is possible that both bounds are violated at the
    % same time. To account for this possibility, both components are
    % checked independently at every step. This is an approximation,
    % because the position prior to the update in the modified 
    % direction is not exactly on the boundary.
    possible_x = x + tdep_fn(t) * dt .* v_half;

    if possible_x(1) < bounds(1) || possible_x(1) > bounds(2)
        v_half(1) = -v_half(1);
    end

    if possible_x(2) < bounds(1) || possible_x(2) > bounds(2)
        v_half(2) = -v_half(2);
    end

    % Update based on the (possibly) changed v_half
    x = x + tdep_fn(t) * dt .* v_half;

    % Acceleration behaves the same way as without reflection at the
    % boundaries.
    a = -1 .* gradient_fn(x(1), x(2))';
    v = v_half + (dt/2) .* a;
    t = t + dt/2;

    if mod(step, cap_frame_every) == 0
        snapshot_times(frame_idx) = t;
        position(frame_idx, :) = x;
        velocity(frame_idx, :) = v;
        observable(frame_idx) = fn_oracle(x(1), x(2));
        frame_idx = frame_idx + 1;
    end

    step = step + 1;
end
end
