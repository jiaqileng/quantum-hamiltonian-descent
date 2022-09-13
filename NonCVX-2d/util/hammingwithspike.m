function [string, costfun] = hammingwithspike(num_bits)

% Input: num_bits = integer. 
% Output: 
% string = a matrix of strings in the hypercube. Each row 
%          is the index of a vertex.
% costfun = a column vector with cost function (32) in 
%         http://arxiv.org/abs/quant-ph/0201031.

if mod(num_bits , 4) ~= 0
    fprintf('num_bits must be a multiple of 4!\n');
    return;
end

N = 2^num_bits;
enumerate = (0:N-1)';
string = de2bi(enumerate,'left-msb');
costfun = zeros(N,1);

for k = 1:N
    w = sum(string(k,:));
    if w == num_bits/4
        costfun(k) = num_bits;
    else
        costfun(k) = w;
    end
end

end