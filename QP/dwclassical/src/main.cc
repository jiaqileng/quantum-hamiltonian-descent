#define EIGEN_NO_DEBUG
#define NDEBUG

#include <algorithm>
#include <array>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>

#include <fstream>
#include <iostream>
#include <sstream>

#include <chrono>


#include <eigen3/Eigen/Dense> // Different platforms may require 'eigen3/' or may not
#include <eigen3/Eigen/SparseCore>

#include "cnpy.h"


using namespace std;


/*
Calculates the difference in energy caused by changing one rotor's orientation.
params:
    cosTheta: collection of cos(rotor angles)
    idx: identifier for the rotor under consideration
    proposedAngle: proposed replacement angle for the rotor under consideration
    h, J: Ising model parameters
    A_s, B_s: Annealing schedule values at the current percent-time (s=t_current/T_final, rather than the simulated time t_current)
returns:
    double, the difference in configuration energy that would result from changing rotor_{idx}'s angle to proposed_theta.
*/
template <typename DerivedTheta, typename Derivedh, typename DerivedJFull>
double delta_E(
    const Eigen::MatrixBase<DerivedTheta>& cosTheta,
    const int& idx,
    const double& proposedAngle,
    const Eigen::MatrixBase<Derivedh>& h,
    const Eigen::SparseMatrixBase<DerivedJFull>& J_full,
    const double& A_s,
    const double& B_s);

/*
Calculates the energy of a configuration.
params:
    cosTheta: collection of cos(rotor angles)
    h, J: Ising model parameters
    A_s, B_s: Annealing schedule values at the current percent-time (s=t_current/T_final, rather than the simulated time t_current)
returns:
    double, the configuration energy.
*/
template <typename DerivedTheta, typename Derivedh, typename DerivedJ>
double hamiltonian(
    const Eigen::MatrixBase<DerivedTheta>& cosTheta,
    const Eigen::MatrixBase<Derivedh>& h,
    const Eigen::SparseMatrixBase<DerivedJ>& J,
    const double& A_s,
    const double& B_s);

double probability_of_acceptance(const double& deltaE, const double& kbT);


int main(int argc, char* argv[])
{
    cout << "Arguments: " << argc << endl;
    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }

    if (argc != 3)
    {
        cout << "Wrong number of args, expecting {start instance} {number of instances to run}" << endl;
        return 0;
    }

    int start_instance = stoi(argv[1]);
    int num_instances = stoi(argv[2]);

    // True randomness, seed with the random_device
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> unitDis(0.0, 1.0);
    uniform_real_distribution<> angleDis(0.0, 2.0*M_PI);


    const string PATH_TO_QP_DIR = "/Users/ethan/LocalResearchData/HamiltonianDescent/TESTDATA/QP/";


    const int PROBLEM_DIMENSION = 50;
    const int BITS_PER_VAR = 8;

    const int QUBO_DIMENSION = PROBLEM_DIMENSION * BITS_PER_VAR;


    // If variables are spread across chains, the configuration will not have predictable
    // size based only on the QP dimension and bits per variable. It can be loaded and computed,
    // but then the optimizers can't optimize? Hard coding it; just be sure only one line is
    // ever active at a time depending on the dimension of the problem you mean to run.
    const int FULL_DIMENSION = 2630;
    // const int FULL_DIMENSION = 3276;
    // const int FULL_DIMENSION = 4064;


    const double KB = 2.083661912e10; // Hz/K
    const double TEMP = 16e-3; // K
    const double KBT = KB * TEMP / 1e9;
    // const double KBT = 0.1;

    const int NUM_SHOTS = 1000;

    const double H_MIN = -4.0;
    const double H_MAX = 4.0;
    const double J_MIN = -1.0;
    const double J_MAX = 1.0;


    // Load s, A, B
    // Same for all instances
    const string PATH_TO_SAB_FILE = PATH_TO_QP_DIR + "/adv6_sAB.npz";

    cnpy::npz_t sAB_npz = cnpy::npz_load(PATH_TO_SAB_FILE);
    double* s_data = sAB_npz["s"].data<double>();
    double* A_data = sAB_npz["A"].data<double>();
    double* B_data = sAB_npz["B"].data<double>();

    array<int, FULL_DIMENSION> shuffledIndices;
    for (int idx = 0; idx < FULL_DIMENSION; ++idx)
    {
        shuffledIndices[idx] = idx;
    }

    Eigen::Matrix<double, FULL_DIMENSION, 1, Eigen::ColMajor, FULL_DIMENSION, 1> h;
    Eigen::SparseMatrix<double, Eigen::RowMajor> J_upper(FULL_DIMENSION, FULL_DIMENSION);
    Eigen::SparseMatrix<double, Eigen::RowMajor> J_full(FULL_DIMENSION, FULL_DIMENSION);

    Eigen::Matrix<double, FULL_DIMENSION, 1, Eigen::ColMajor, FULL_DIMENSION, 1> cosTheta;

    // Vector for shot output converted from cos(angles) to binary QUBO values
    Eigen::Matrix<double, 1, FULL_DIMENSION, Eigen::RowMajor, 1, FULL_DIMENSION> spins;

    // Store reduction from chains to QUBO values
    Eigen::Matrix<double, 1, QUBO_DIMENSION, Eigen::RowMajor, 1, QUBO_DIMENSION> qubo_variables;

    // Vector for the shot output converted from QUBO settings to positions using the Hamming precision vector
    Eigen::Matrix<double, 1, PROBLEM_DIMENSION, Eigen::RowMajor, 1, PROBLEM_DIMENSION> positions;

    // Result matrix; each row is filled with the output of one shot
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> shotResults(NUM_SHOTS, PROBLEM_DIMENSION);


    // Load the chains from the text file
    const string PATH_TO_CHAIN_FILE = PATH_TO_QP_DIR + "/QP-" + to_string(PROBLEM_DIMENSION) + "d-5s/chains" + to_string(PROBLEM_DIMENSION) + ".txt";

    vector<vector<int>> chains;
    ifstream chainIFS(PATH_TO_CHAIN_FILE, ifstream::in);
    string chain_str;

    while (getline(chainIFS, chain_str))
    {
        vector<int> chain;
        istringstream iss(chain_str);
        int site;
        while (iss >> site)
        {
            chain.push_back(site);
        }

        chains.push_back(chain);
    }


    for (int instance = start_instance; instance < start_instance + num_instances; ++instance)
    {
        const string PATH_TO_INSTANCE_DIR = PATH_TO_QP_DIR + "/QP-" + to_string(PROBLEM_DIMENSION) +"d-5s/instance_" + to_string(instance);


        double hMin;
        double hMax;
        double JMin = J_MAX;
        double JMax = J_MIN;

        // Load h
        cnpy::NpyArray h_load = cnpy::npy_load(PATH_TO_INSTANCE_DIR + "/instance_" + to_string(instance) + "_h.npy");
        double* h_data = h_load.data<double>();
        for (int i = 0; i < h_load.shape[0]; ++i)
        {
            h[i] = h_data[i];
        }

        hMin = h.minCoeff();
        hMax = h.maxCoeff();


        // Reset and load J
        J_upper.setZero();

        cnpy::npz_t J_coo_npz = cnpy::npz_load(PATH_TO_INSTANCE_DIR + "/instance_" + to_string(instance) + "_J.npz");
        uint32_t* J_rows = J_coo_npz["row"].data<uint32_t>();
        uint32_t* J_cols = J_coo_npz["col"].data<uint32_t>();
        double* J_data = J_coo_npz["data"].data<double>();

        J_upper.reserve(J_coo_npz["data"].shape[0]);
        for (int i = 0; i < J_coo_npz["data"].shape[0]; ++i)
        {
            J_upper.insert(J_rows[i], J_cols[i]) = J_data[i];

            // Eigen's sparse matrix API doesn't have minCoeff or maxCoeff, so
            // just track them as J is loaded
            if (J_data[i] < JMin)
            {
                JMin = J_data[i];
            }
            if (J_data[i] > JMax)
            {
                JMax = J_data[i];
            }
        }

        J_upper.makeCompressed();

        // Use the h, J problem range and allowed range to determine the largest
        // possible rescale factor
        double rescaleFactor = 0;
        rescaleFactor = max(rescaleFactor, max(hMax/H_MAX, 0.0));
        rescaleFactor = max(rescaleFactor, max(hMin/H_MIN, 0.0));
        rescaleFactor = max(rescaleFactor, max(JMax/J_MAX, 0.0));
        rescaleFactor = max(rescaleFactor, max(JMin/J_MIN, 0.0));

        // cout << 'd' << PROBLEM_DIMENSION << " instance " << instance << " rescaled by factor of 1/" << rescaleFactor << '=' << 1.0/rescaleFactor << endl;

        h /= rescaleFactor;
        J_upper /= rescaleFactor;

        J_full = J_upper.selfadjointView<Eigen::Upper>();
        J_full.makeCompressed();

        int shotIdx = 0;

        // Time, in ns
        double t;

        // End time of simulation
        double t_f = 800e3; // ns
        //double t_f = 20e3; // ns

        double num_steps = 1e2;

        // Time step between flip trials
        double dt = t_f / num_steps; // ns


        // Defining the schedule
        // [us, s]
        // [[0,0],[400, 0.3],[640, 0.6],[800,1]]

        // Schedule defined in terms of a set of increasing points (time, s)
        // s is linearly interpolated in time between the values specified
        vector<tuple<double, double>> schedule;
        schedule.push_back(tuple<double, double>(0.0, 0.0));

        schedule.push_back(tuple<double, double>(400e3, 0.3));
        schedule.push_back(tuple<double, double>(640e3, 0.6));
        schedule.push_back(tuple<double, double>(800e3, 1.0));

        //schedule.push_back(tuple<double, double>(20e3, 1.0));

        int schedulePhaseIdx;
        int annealSegmentIdx;
        double annealSegmentStartTime;
        double annealSegmentEndTime;
        double annealSegmentDuration;
        double s_annealSegmentStart;
        double s_annealSegmentEnd;
        double s_annealSegmentRange;
        double interpFrac;
        double s;
        double A_t;
        double B_t;

        double proposedAngle;
        double dE;

        while (shotIdx < NUM_SHOTS)
        {
            cout << "i " << instance << " shot " << shotIdx << endl;

            // Generate a random angle on [0, 2*PI) to start each shot with on each rotor
            // Use uniform random angles because the ground state at the beginning of annealing is the uniform superposition, suggesting a classical analogue is initializing with an equal likelihood for any angle.
            for (int idx = 0; idx < FULL_DIMENSION; ++idx)
            {
                cosTheta[idx] = angleDis(gen);
            }

            // Almost all calculations require the cosine of theta.
            cosTheta = cosTheta.array().cos();

            t = 0;

            annealSegmentIdx = 0;
            schedulePhaseIdx = 0;

            annealSegmentStartTime = get<0>(schedule[0]);
            annealSegmentEndTime = get<0>(schedule[1]);
            annealSegmentDuration = annealSegmentEndTime - annealSegmentStartTime;

            s_annealSegmentStart = get<1>(schedule[0]);
            s_annealSegmentEnd = get<1>(schedule[1]);
            s_annealSegmentRange = s_annealSegmentEnd - s_annealSegmentStart;

            A_t = A_data[0];
            B_t = B_data[0];

            // auto start = chrono::high_resolution_clock::now();
            while (t < t_f)
            {
                if (t >= get<0>(schedule[annealSegmentIdx+1]))
                {
                    ++annealSegmentIdx;

                    annealSegmentStartTime = get<0>(schedule[annealSegmentIdx]);
                    annealSegmentEndTime = get<0>(schedule[annealSegmentIdx+1]);
                    annealSegmentDuration = annealSegmentEndTime - annealSegmentStartTime;

                    s_annealSegmentStart = get<1>(schedule[annealSegmentIdx]);
                    s_annealSegmentEnd = get<1>(schedule[annealSegmentIdx+1]);
                    s_annealSegmentRange = s_annealSegmentEnd - s_annealSegmentStart;
                }

                // Fractional amount into this anneal schedule (line) segment
                interpFrac = (t - annealSegmentStartTime) / annealSegmentDuration;

                // Fractional amount as a coefficient on the difference in s
                // between the endpoints of the line segment
                s = s_annealSegmentStart + interpFrac * s_annealSegmentRange;

                // Use the same index until s meets or exceeds the time percentage of the next phase of the annealing schedule
                if (s >= s_data[schedulePhaseIdx+1])
                {
                    ++schedulePhaseIdx;

                    A_t = A_data[schedulePhaseIdx];
                    B_t = B_data[schedulePhaseIdx];

                }

                /* Skip the work and check that the schedule is working
                if ((int) t / 10000 == t / 10000.0)
                {
                    cout << t << ' ' << s << ' ' << A_t << ' ' << B_t << endl;
                }
                t += dt;
                continue;
                */

                // Randomize the order in which flips are attempted
                shuffle(shuffledIndices.begin(), shuffledIndices.end(), gen);

                for (int& trialIdx: shuffledIndices)
                {
                    proposedAngle = angleDis(gen);

                    dE = delta_E(cosTheta,
                                 trialIdx,
                                 proposedAngle,
                                 h, J_full,
                                 A_t, B_t);

                    if (unitDis(gen) < probability_of_acceptance(dE, KBT))
                    {
                        // double before = hamiltonian(cosTheta, h, J_upper, A_t, B_t);
                        cosTheta[trialIdx] = cos(proposedAngle);
                        // double after = hamiltonian(cosTheta, h, J_upper, A_t, B_t);
                        // cout << dE << " =? " << after-before << endl;
                    }
                }

                t += dt;
            }

            // auto end = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsed = end-start;
            // cout << "dim " << FULL_DIMENSION << " time elapsed: " << elapsed.count() << endl;


            for (int i = 0; i < FULL_DIMENSION; ++i)
            {
                // QUBO <==> Ising maps by:
                // (2*QUBO - 1) = ISING
                // (ISING + 1) / 2 = QUBO

                // Snap negative to -1, positive to +1
                if (cosTheta[i] <= 0)
                {
                    spins[i] = 0.0;
                } else {
                    spins[i] = 1.0;
                }
            }

            for (int chainIdx = 0; chainIdx < QUBO_DIMENSION; ++chainIdx)
            {
                vector<int> chain = chains[chainIdx];
                double acc = 0.0;

                for (int site : chain)
                {
                    acc += spins[site];
                }

                if (acc > chain.size() / 2.0)
                {
                    qubo_variables[chainIdx] = 1;
                } else if (acc < chain.size() / 2.0)
                {
                    qubo_variables[chainIdx] = 0;
                } else {
                    qubo_variables[chainIdx] = round(unitDis(gen));
                }
            }


            for (int d = 0; d < PROBLEM_DIMENSION; ++d)
            {
                positions[d] = qubo_variables.segment<BITS_PER_VAR>(BITS_PER_VAR * d).sum() / ((double) BITS_PER_VAR);
            }

            shotResults.row(shotIdx) = positions;

            ++shotIdx;
        }


        cout << "finished instance " << instance << endl;

        string save_target_fname = PATH_TO_INSTANCE_DIR + "/instance_" + to_string(instance) + "_scaled_timed_1e2_sweeps_temp_16e-3K_results.npy";

        cnpy::npy_save(
            save_target_fname,
            shotResults.data(),
            vector<size_t> {(uint32_t) shotResults.rows(), (uint32_t) shotResults.cols()}
        );
    }

    return 0;
}


// If the energy would decrease, accept the change.
// If the energy would increase, accept the change according to
// the Boltzmann distribution set by the temperature of simulation
double probability_of_acceptance(const double& deltaE, const double& kbT)
{
    return min(1.0, exp(-deltaE / kbT));
}


template <typename DerivedTheta, typename Derivedh, typename DerivedJFull>
double delta_E(
    const Eigen::MatrixBase<DerivedTheta>& cosTheta,
    const int& idx,
    const double& proposedAngle,
    const Eigen::MatrixBase<Derivedh>& h,
    const Eigen::SparseMatrixBase<DerivedJFull>& J_full,
    const double& A_s,
    const double& B_s)
{
    double driving_term;
    double problem_term;

    driving_term = -0.5 * A_s * (sin(proposedAngle) - sin(acos(cosTheta[idx])));
    problem_term = 0.5 * B_s * (h[idx] + J_full.row(idx).dot(cosTheta)) * (cos(proposedAngle) - cosTheta[idx]);

    return driving_term + problem_term;
}


template <typename DerivedTheta, typename Derivedh, typename DerivedJ>
double hamiltonian(
    const Eigen::MatrixBase<DerivedTheta>& cosTheta,
    const Eigen::MatrixBase<Derivedh>& h,
    const Eigen::SparseMatrixBase<DerivedJ>& J,
    const double& A_s,
    const double& B_s)
{
    double driving_term;
    double problem_term;

    driving_term = -0.5 * A_s * cosTheta.array().acos().sin().sum();
    problem_term = 0.5 * B_s * (h.transpose().dot(cosTheta) + cosTheta.transpose().dot(J * cosTheta));

    return driving_term + problem_term;
}
