function [expectation] = positionDependentExpectation(wave, op_vals)
%positionDependentExpectation Approximates the expectation of op_vals for a
% wave.

% Input
% wave: normalized complex wavefunction as a row vector
% op_vals: scalar value for each point in the region under consideration
%   as a column vector

% Output
% expectation: scalar value of expectation.
    prob = conj(wave) .* wave;    
    expectation = prob * op_vals;
end

