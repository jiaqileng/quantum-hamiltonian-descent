function [indep_vals, dep_vals] = filterMinima(indep, dep)
%filterMinima Filters a vector for all of its local maxima

len = numel(indep);
write_idx = 1;

indep_vals = zeros(1,1);
dep_vals = zeros(1,1);

% Ends of array are edgecases for a filter that consumes 3 elements as
% input.
if indep(1) < indep(2)
    indep_vals(write_idx) = indep(1);
    dep_vals(write_idx) = dep(1);
    write_idx = write_idx + 1;
end


for i = 2:(len-1)
    if indep(i) < indep(i-1) && indep(i) < indep(i+1)
        indep_vals(write_idx) = indep(i);
        dep_vals(write_idx) = dep(i);
        write_idx = write_idx + 1;
    end
end

% Computing the slope breaks when we have this long term value for the
% expected potential.
% Take the endpoint if it's very close to the prior element (if the value
% has essentially converged; it should have when applying this analysis)
% if indep(len) <= indep(len-1)
%     indep_vals(write_idx) = indep(len);
%     dep_vals(write_idx) = dep(len);
% end


end

