#include <cmath>
#include <complex>
#include <vector>
// #define EIGEN_RUNTIME_NO_MALLOC // Enable runtime tests for allocations

#include <eigen3/Eigen/Eigen>

#include <iostream>
#include <chrono>

using namespace std::complex_literals;


void scheq_leapfrog(const Eigen::VectorXcd& PSI_0,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor>& H_T,
                    const Eigen::VectorXd& H_V,
                    std::vector<Eigen::VectorXcd>& out,
                    double (*phi)(double&, double&),
                    double gamma,
                    double dt,
                    double end_time,
                    int num_checkpoints)
{

    // Validate the number of checkpoints to make, num_checkpoints.
    // One checkpoint will only save the final state.
    // Two checkpoints will save the initial and final states.
    // Otherwise, checkpoints are saved at evenly spaced intervals over the
    // number of time steps.
    if (num_checkpoints < 1)
    {
        throw std::invalid_argument("num_checkpoints is less than 1");
    }

    // Count the number of steps taken
    int step = 0;

    // Track the simulation time so it can be written as part of the output
    double t = 0;

    // Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> H_T = H_T;
    // Eigen::VectorXcd H_V = H_V;

    Eigen::VectorXd re = PSI_0.real();
    Eigen::VectorXd im = (-1i * PSI_0.imag()).real();

    // Index into output to write state snapshot
    int out_idx = 0;

    int num_steps = end_time / dt;

    // Ensure that the endtime is a multiple of dt by taking the next smaller
    // possible dt. Also update num_steps.
    if (end_time / num_steps != dt)
    {
        dt = end_time / (++num_steps);
    }

    // Default value will only save the final state.
    int steps_per_checkpoint = num_steps;

    bool append_final_state = true;

    if (num_checkpoints > 1)
    {
        std::cout << "Checkpoint for step " << step << " at sim time " << t << std::endl;
        out[out_idx++] = PSI_0;

        // Divide up the time range into num_checkpoints-1 steps
        steps_per_checkpoint = std::floor(num_steps / (num_checkpoints - 1));

        // Do not append the state if the last checkpoint already lands on the end
        if (num_steps % steps_per_checkpoint == 0)
        {
            append_final_state = false;
        }
    }

    // Reduced dt used to take half steps.
    double rdt = 0.5 * dt;

    double kinetic_time_dep;

    // Profile the main loop runtime.
    auto t1 = std::chrono::high_resolution_clock::now();

    // Runtime heap allocation check for temp object creation.
    // Eigen::internal::set_is_malloc_allowed(false);
    while (step < num_steps)
    {
        // Update p by d/dt(p) = -Hq with half step rdt
        // kinetic_time_dep = phi(t, gamma);
        re.noalias() += (rdt * kinetic_time_dep) * (H_T * im);
        re.noalias() += rdt * H_V.cwiseProduct(im);

        t += rdt;

        // Update q by d/dt(q) = Hp with full step dt
        kinetic_time_dep = phi(t, gamma);
        im.noalias() -= (dt * kinetic_time_dep) * (H_T * re);
        im.noalias() -= dt * H_V.cwiseProduct(re);

        // t += rdt;

        // Update p by d/dt(p) = -Hq at t+rdt with half step rdt
        kinetic_time_dep = phi(t, gamma);
        re.noalias() += (rdt * kinetic_time_dep) * (H_T * im);
        re.noalias() += rdt * H_V.cwiseProduct(im);

        t += rdt;
        ++step;

        if (step % steps_per_checkpoint == 0)
        {
            std::cout << "Checkpoint for step " << step << " at sim time " << t << std::endl;
            // std::cout << re.head(10) + 1i*im.head(10) << std::endl << std::endl << std::endl;
            out[out_idx++] = re + 1i*im;
        }
    }
    // Eigen::internal::set_is_malloc_allowed(true);

    if (append_final_state)
    {
        std::cout << "Checkpoint for step " << step << " at sim time " << t << std::endl;
        out.push_back(re + 1i*im);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Main loop time: " << duration << "e-6 s"  << std::endl;
}
