function [F,G] = QPuserfun(x, Q, b, Q_c)
% Objective and constraints
F = [(transpose(x) * Q * x) / 2 + dot(b, x) ; Q_c * x];
% Jacobian of nonlinear part of F
G = [transpose(Q*x) + transpose(b) ; Q_c];
G = transpose(G(1,:));