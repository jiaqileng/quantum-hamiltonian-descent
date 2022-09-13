#ifndef SCHEQLEAPFROG_H
#define SCHEQLEAPFROG_H

void scheq_leapfrog(const Eigen::VectorXcd& PSI_0,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor>& H_T,
                    const Eigen::VectorXd& H_V,
                    std::vector<Eigen::VectorXcd>& out,
                    double (*phi)(double&, double&),
                    double gamma,
                    double dt,
                    double end_time,
                    int num_checkpoints);

#endif
