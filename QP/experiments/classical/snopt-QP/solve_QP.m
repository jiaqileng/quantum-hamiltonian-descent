function [x, F, info] = solve_QP(Q, b, Q_c, b_c, instance, x_0)
%SOLVE_QP Summary of this function goes here
%   Detailed explanation goes here

options.printfile = '';
options.specsfile = which(strcat('instance_', int2str(instance), '.spc'));
options.system_information ='yes';
options.name = strcat('instance_', int2str(instance));
options.start = 'Warm';

ObjAdd = 0;
ObjRow = 1;
% objective + number of constraints
nF = 1 + size(Q_c,1);
% problem dimension
n = size(Q, 1);

xlow = zeros(n, 1);
xupp = ones(n, 1);
xmul = zeros(n,1);
xstate = zeros(n,1);

Flow = [-Inf ; b_c];
Fupp = [Inf ; b_c];
Fmul = zeros(nF,1);
Fstate = zeros(nF,1);
%[~, G] = QPuserfun(x_0, Q, b, Q_c);
G.row = ones(n, 1);
G.col = transpose(1:n);
num_constraints = size(Q_c,1);
dimension = size(Q_c,2);
iAfun = zeros(size(Q_c,1)*size(Q_c,2),1);
jAvar = zeros(size(Q_c,1)*size(Q_c,2),1);
for i=1:num_constraints
    for j=1:dimension
        iAfun((i-1) * num_constraints + j) = i+1;
        jAvar((i-1) * num_constraints + j) = j;
    end
end
A.row = iAfun(:,1);
A.col = jAvar(:,1);
A.val = reshape(Q_c.',1,[]);
%A = zeros(size(G));
[x,F,info] = snopt(x_0, xlow, xupp, xmul, xstate,  ...
		  Flow, Fupp, Fmul, Fstate,     ...
		  @(x)QPuserfun(x, Q, b, Q_c), ObjAdd, ObjRow, ...
		  A, G, options);

end

