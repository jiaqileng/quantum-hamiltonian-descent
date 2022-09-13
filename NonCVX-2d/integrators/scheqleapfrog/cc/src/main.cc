#include <iostream>
#include <vector>

#include <complex>
#include <cmath>

#include <eigen3/Eigen/Eigen>

#include "scheqleapfrog.h"
#include "scheqleapfrogio.h"

#include <chrono>

using namespace std::complex_literals;

double phi(double& t, double& gamma);

double phi(double& t, double& gamma)
{
    return std::exp(-gamma * t);
}

// double phi(double& t, double& gamma)
// {
//     return std::exp(-gamma * std::sqrt(t));
// }

// double phi(double& t, double& gamma)
// {
//     return 1.0/(1.0 + gamma * t);
// }


int main()
{
    // Basic tests for IO (to be moved)
    // Eigen::VectorXcd test_psi;
    // read_initial_state("testinput.txt", test_psi);
    // std::cout << test_psi << std::endl;

    // std::vector<Eigen::VectorXcd> snapshots(4);
    // Eigen::VectorXcd v(4);
    // for (int i = 0; i <= 3; ++i)
    // {
    //     for (int j = 0; j <= 3; ++j)
    //     {
    //         v(j) = (double)(i*4+j) + 1i*(double)j;
    //     }
    //     snapshots[i] = v;
    // }
    //
    // for (auto& state : snapshots)
    // {
    //     std::cout << state << std::endl;
    // }
    //
    // write_state_snapshots(snapshots, "testoutput.txt");

    // Eigen::SparseMatrix<double, Eigen::RowMajor> H_T;
    //
    // read_laplacian("lap2d.txt", H_T);
    //
    // std::cout << H_T << std::endl;

    // High resolution timer for different sections of the code.
    auto t1 = std::chrono::high_resolution_clock::now();

    // Experimental params
    double L = 10;
    int num_dims = 2;
    int cells_per_dim = 150;
    int num_cells = std::pow(cells_per_dim, num_dims);
    double step_size = L / cells_per_dim;


    /* Square well
    double step_size_1d = L / (cells_per_dim + 1);

    // Have the domain here so it can be written. Maybe we can do a 1d domain for
    // each dimension, or leave it as n dimensions of the same domain.
    Eigen::VectorXd dom = Eigen::VectorXd::LinSpaced(num_cells, step_size, L-step_size);

    // Construct sparse 1D Laplacian from triplets
    Eigen::SparseMatrix<double, Eigen::RowMajor> H_T(num_cells, num_cells);
    H_T.reserve(Eigen::VectorXi::Constant(num_cells, 3));

    std::vector<Eigen::Triplet<double>> lap1d_elems;
    for (int idx = 0; idx < num_cells; ++idx)
    {
        if (idx > 0)
        {
            lap1d_elems.push_back(Eigen::Triplet<double>(idx, idx-1, 1));
        }

        lap1d_elems.push_back(Eigen::Triplet<double>(idx, idx, -2));

        if (idx < num_cells-1)
        {
            lap1d_elems.push_back(Eigen::Triplet<double>(idx, idx+1, 1));
        }
    }

    H_T.setFromTriplets(lap1d_elems.begin(), lap1d_elems.end());
    H_T.makeCompressed();

    H_T *= -0.5 * std::pow(step_size, -2);

    // Square well potential has value 0 everywhere in the well.
    Eigen::VectorXd H_V = Eigen::VectorXd::Zero(num_cells);

    // Eigen::VectorXcd PSI_0 = Eigen::VectorXcd(num_cells);
    // for (int idx = 0; idx < num_cells; ++idx)
    // {
    //     PSI_0(idx) = 1.0/std::sqrt(14) * std::sqrt(2 / L) * std::sin(M_PI * dom(idx) / L)
    //         + 2.0/std::sqrt(14) * std::sqrt(2 / L) * std::sin(2 * M_PI * dom(idx) / L)
    //         + 3.0/std::sqrt(14) * std::sqrt(2 / L) * std::sin(3 * M_PI * dom(idx) / L);
    // }

    // Squared for 1d
    // dt = 0.95 * std::pow(step_size, 2);

    */ // END SQUARE WELL-SPECIFIC


    // Have the domain here so it can be written. Maybe we can do a 1d domain for
    // each dimension, or leave it as n dimensions of the same domain.
    Eigen::VectorXd dom_2d = Eigen::VectorXd::LinSpaced(cells_per_dim, -L/2, L/2);

    Eigen::SparseMatrix<double, Eigen::RowMajor> H_T;
    read_laplacian("lap2d.txt", H_T);
    H_T *= -0.5 * std::pow(step_size, -2);

    // Styblinsky Tang Function as potential
    double x;
    double y;
    int target_idx;
    Eigen::VectorXd H_V = Eigen::VectorXd(num_cells);

    for (int y_idx = cells_per_dim-1; y_idx > 0; --y_idx)
    {
        for (int x_idx = 0; x_idx < cells_per_dim; ++x_idx)
        {
            x = dom_2d(x_idx);
            y = dom_2d(y_idx);
            target_idx = (cells_per_dim-1 - y_idx) * cells_per_dim + x_idx;
            H_V(target_idx) = 0.5 * (std::pow(x, 4) - (16 * std::pow(x, 2)) + (5 * x)
                                     + std::pow(y, 4) - (16 * std::pow(y, 2)) + (5 * y));
        }
    }

    Eigen::VectorXcd PSI_0;
    read_initial_state("gaussian.txt", PSI_0);


    double gamma = 0.25;

    double dt;

    // Step size cubed for stability with 2d domain
    // Must be slightly less than 1 times the cube; choose 0.95 as an arbitrary
    // slightly smaller constant.
    dt = 0.95 * std::pow(step_size, 3);
    dt = 0.0003;

    std::cout << "dt: " << dt << std::endl;

    double T = 30;
    // It may be possible to scale dt and the simulation end time up and H_T down.
    // This could give numerical stability back to dimensions d where step_size^d
    // is prohibitively small and contents of H_T are prohibitively large.
    double end_time = T;

    // Number of checkpoints to take (excluding the initial state)
    // The final state may not be equally spaced from the other checkpoints.
    int num_checkpoints = 20;

    std::vector<Eigen::VectorXcd> output(++num_checkpoints, Eigen::VectorXcd(num_cells));


    // Time the leapfrog method runtime.
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Setup time: " << duration << "e-6 s"  << std::endl;

    t1 = std::chrono::high_resolution_clock::now();

    scheq_leapfrog(
        PSI_0,
        H_T,
        H_V,
        output,
        &phi,
        gamma,
        dt,
        end_time,
        num_checkpoints
    );

    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Leapfrog runtime: " << duration << "e-6 s" << std::endl << std::endl;

    write_state_snapshots(output, "evolution.txt");
}
