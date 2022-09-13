function [L] = laplacian1d(total_num_cells)
%LAPLACIAN1D Builds a sparse matrix for the Laplacian operator in one
% dimension. 

e = ones(total_num_cells, 1);
L = spdiags([e -2*e e], -1:1, total_num_cells, total_num_cells);
end