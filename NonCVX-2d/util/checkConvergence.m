function converged = checkConvergence(observable_vals, ...
                        lookback_offsets, ...
                        percent_diff)
%CHECKCONVERGENCE Determine whether an observable has converged for an
%experiment under dissipative evolution by percent difference.
%   Evaluate the convergence of an observable from observed snapshots 
%   stored in observable_vals. Snapshots to consider to determine 
%   convergence are specified as positive integers in lookback_offsets to 
%   look back n snapshots from the end of psi.

% Default return value: the wave has converged. To be overwritten if 
% evidence is found to the contrary.
converged = 1;

current_fn_val = observable_vals(end);

for offset = lookback_offsets
    lookback_fn_val = observable_vals(end-offset);
    
    % If any of the waves at a lookback offset are outside of tol
    % from the most recent wave, the value has not converged.
    if abs(current_fn_val - lookback_fn_val) / abs(current_fn_val) > percent_diff
        converged = 0;
        break;
    end
end

end
