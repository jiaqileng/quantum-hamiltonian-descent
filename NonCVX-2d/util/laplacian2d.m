function [L] = laplacian2d(total_num_cells)
%LAPLACIAN2D Builds a sparse matrix for the Laplacian operator in two
% dimensions. 
e = ones(total_num_cells, 1);
B = spdiags([e -4*e e], -1:1, total_num_cells, total_num_cells);
A = spdiags([e 0*e e],  -1:1, total_num_cells, total_num_cells);
I = speye(total_num_cells);
L = kron(I,B) + kron(A,I);
end

