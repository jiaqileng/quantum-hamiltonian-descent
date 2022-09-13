#ifndef SCHEQLEAPFROGIO_H
#define SCHEQLEAPFROGIO_H

void read_initial_state(const char* input_fname, Eigen::VectorXcd& psi);

void read_laplacian(const char* input_fname, Eigen::SparseMatrix<double, Eigen::RowMajor>& laplacian);

void write_state_snapshots(std::vector<Eigen::VectorXcd>& snapshots, const char* output_fname);

#endif
