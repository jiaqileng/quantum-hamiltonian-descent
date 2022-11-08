#include <iostream>
#include <complex>
#include <math.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include "to_npy.hpp"
#include <omp.h>
#include <matio.h>
#include <filesystem>

#define I complex<double>(0, 1)

using namespace std::filesystem;
using namespace Eigen;

// For the first NUMBER_OF_ITERATIONS_A iterations, save wave function every CAPTURE_FRAME_EVERY_A iterations.
// Afterwards, save the wave function every CAPTURE_FRAME_EVERY_B iterations.
const int NUMBER_OF_ITERATIONS_A = 1000;
const int CAPTURE_FRAME_EVERY_A = 100; 
const int CAPTURE_FRAME_EVERY_B = 500;
const int NUMBER_OF_FUNCTIONS = 22;
const int RANK = 2;
const string HOME_DIR = getenv("HOME");
const string DATA_DIR = path(HOME_DIR).append("QHD_DATA");
const string NONCVX_DIR = path(DATA_DIR).append("NonCVX-2d");

double t_dep_1(const double &t) {
    return 2 / (1e-3 + t * t * t);
}
double t_dep_2(const double &t) {
    return 2 * t * t * t;
}

void initialize_psi(ArrayXXcd &y, const double &L, const int &N) {
    double x1, x2;
    int i, j;
    double sum = 0;
    double stepsize = 2 * L / N;

    #pragma omp parallel for private(i, j, x1, x2) reduction(+: sum) schedule(static)
    for (i = 0; i < N; i++) {
        x1 = -L + i * stepsize;
        for (j = 0; j < N; j++) {
            x2 = -L + j * stepsize;
            // transpose to be consistent with matlab
            y(j, i) = (double) 1 / N;
            sum += (y(j, i) * conj(y(j, i))).real();
        }
    }
}

void load_potential(ArrayXXd &V, const int &N, const int &function_id, double *potential_data) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // matlab is column major order
            V(i, j) = potential_data[NUMBER_OF_FUNCTIONS * (j * N + i) + function_id];
        }
    }
}
void initialize_kinetic_operator(ArrayXXd &y, const double &L, const int &N)
{

    int freq[N];
    int i, j, x1, x2;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < N / 2; i++) {
        freq[i] = i;
        freq[N / 2 + i] = -N / 2 + i;
    }

    #pragma omp parallel for private(i, j, x1, x2) schedule(static)
    for (i = 0; i < N; i++) {
        x1 = freq[i];
        for (j = 0; j < N; j++) {
            x2 = freq[j];
            y(i, j) = (0.5 * M_PI * M_PI * (x1 * x1 + x2 * x2)) / (L * L);
        }
    }
}

void hadamard_product(ArrayXXcd &A, ArrayXXcd &B, ArrayXXcd &out, const int &N) {
    int i, j;
    #pragma omp parallel for private(i,j) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            out(i,j) = A(i,j) * B(i,j);
        }
    }
}

void pseudospec(const ArrayXXd &V, const double &L, const int &num_cells, ArrayXXcd &psi,
                const double &dt, const double &T, const string function_name)
{
    /*
    *  Pseudospectral solver for time-dependent Schrodinger equation
    *    H(t) = -0.5 * tdep(t) * Laplacian + V(x)
    *  defined on [-L,L]^2
    */

    string function_dir = path(NONCVX_DIR).append(function_name);
    string wfn_dir = path(function_dir).append(function_name + "_QHD" + to_string(num_cells) + "_WFN");
    if (!filesystem::exists(function_dir)) {
        filesystem::create_directory(function_dir);
    }
    if (!filesystem::exists(wfn_dir)) {
        filesystem::create_directory(wfn_dir);
    }


    int num_steps = T / dt;
    double t = 0;
    ArrayXd snapshot_times(num_steps+1);
    ArrayXd expected_potential(num_steps+1);
    ArrayXXd kinetic_operator(num_cells, num_cells);
    ArrayXXcd u(num_cells, num_cells);
    ArrayXXcd psi_new(num_cells, num_cells);
    ArrayXXcd temp(num_cells, num_cells);
    int i, j;
    int n[RANK];
    for (i = 0; i < RANK; i++)
    {
        n[i] = num_cells;
    }
    
    // Save wavefunction at time 0
    write_complex_to_npy(psi, path(wfn_dir).append("psi_0.npy"));

    // Initialize kinetic operator
    initialize_kinetic_operator(kinetic_operator, L, num_cells);

    // Compute expected potential
    double expected_value = 0;
    #pragma omp parallel for private(i,j) reduction(+: expected_value)
    for (i = 0; i < num_cells; i++) {
        for (j = 0; j < num_cells; j++) {
            expected_value += (psi(i,j) * conj(psi(i,j))).real() * V(i,j);
        }
    }
    snapshot_times(0) = t;
    expected_potential(0) = expected_value;

    // Setup for fftw
    fftw_complex *in;
    fftw_complex *out;
    fftw_plan p1;
    fftw_plan p2;

    if (fftw_init_threads() == 0)
    {
        cout << "Error with fftw_init_threads." << endl;
        exit(EXIT_FAILURE);
    }
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    // Used to compute fft
    in = (fftw_complex *)&u(0, 0);
    out = (fftw_complex *)&u(0, 0);
    p1 = fftw_plan_dft(RANK, n, in, out, FFTW_FORWARD, FFTW_MEASURE);

    // Used to compute ifft
    in = (fftw_complex *)&u(0, 0);
    out = (fftw_complex *)&psi_new(0, 0);
    p2 = fftw_plan_dft(RANK, n, in, out, FFTW_BACKWARD, FFTW_MEASURE);

    // Timing
    double time1 = 0, time2 = 0;
    double start, end;
    start = omp_get_wtime();
    for (int step = 1; step <= num_steps; step++) {

        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < num_cells; i++) {
            for (j = 0; j < num_cells; j++) {
                temp(i,j) = exp(- I * dt * t_dep_2(t) * V(i,j));
            }
        }
        hadamard_product(temp, psi, u, num_cells);

        fftw_execute(p1);

        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < num_cells; i++) {
            for (j = 0; j < num_cells; j++) {
                temp(i,j) = exp(- I * dt * t_dep_1(t) * kinetic_operator(i,j));
            }
        }
        hadamard_product(temp, u, u, num_cells);

        // ifft2
        fftw_execute(p2);

        // Normalize
        double k = pow(num_cells, RANK);
        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < num_cells; i++) {
            for (j = 0; j < num_cells; j++) {
                psi(i,j) = psi_new(i,j) / k;
            }
        }

        t = step * dt;

        if ((step <= NUMBER_OF_ITERATIONS_A) && (step % CAPTURE_FRAME_EVERY_A == 0) || 
            (step > NUMBER_OF_ITERATIONS_A) && (step % CAPTURE_FRAME_EVERY_B == 0)) {
            string str_t = to_string((int) (t * 10));
            string filename = "psi_" + str_t + "e-01.npy";
            write_complex_to_npy(psi, path(wfn_dir).append(filename));
        } 

        // Compute expected potential
        expected_value = 0;
        #pragma omp parallel for private(i,j) reduction(+: expected_value)
        for (i = 0; i < num_cells; i++) {
            for (j = 0; j < num_cells; j++) {
                expected_value += (psi(i,j) * conj(psi(i,j))).real() * V(i,j);
            }
        }
        snapshot_times(step) = t;
        expected_potential(step) = expected_value;

        printf("\r%d / %d", step, num_steps);
        fflush(stdout);
    }
    end = omp_get_wtime();
    printf("\n");
    printf("Pseudospectral integrator runtime = %.5f s\n", end - start);
    
    // Save snapshot times
    string snapshot_times_filename = "snapshot_times_" + function_name + ".npy";
    write_real_to_npy_1d(snapshot_times, path(function_dir).append(snapshot_times_filename));
    // Save expected potential
    string expected_potential_filename = "expected_potential_" + function_name + ".npy";
    write_real_to_npy_1d(expected_potential, path(function_dir).append(expected_potential_filename));
    // Cleanup for fftw
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_cleanup_threads();
}

int main(int argc, char **argv)
{

    if (argc != 6) {
        perror("Expected arguments: ./2d-pseudospec <L> <num_cells> <T> <dt> <potentials_filename>");
        exit(EXIT_FAILURE);
    }
    const double L = stod(argv[1]);
    const int num_cells = stoi(argv[2]);
    const double T = stod(argv[3]);
    const double dt = stod(argv[4]);
    printf("L=%f, num_cells=%d\n", L, num_cells);
    printf("T=%f, dt=%f\n", T, dt);
    
    const char* potentials_filename;

    if (argc == 6) {
        potentials_filename = argv[5];
        cout << "Potentials filename: " << potentials_filename << endl;
    }

    double stepsize = 2 * L / num_cells;
    printf("stepsize=%f\n", stepsize);

    printf("Max threads: %d\n", omp_get_num_procs());
    printf("Threads: %d\n", omp_get_max_threads());

    if (!filesystem::exists(DATA_DIR)) {
        filesystem::create_directory(DATA_DIR);
    }
    if (!filesystem::exists(NONCVX_DIR)) {
        filesystem::create_directory(NONCVX_DIR);
    }
    
    // psi
    ArrayXXcd psi(num_cells, num_cells);

    // potential V
    ArrayXXd V(num_cells, num_cells);

    // matio
    mat_t *matfp;
    matfp = Mat_Open(potentials_filename, MAT_ACC_RDONLY);
    if (matfp == NULL) {
        cerr << "Error opening MAT file " << potentials_filename << endl;
        exit(EXIT_FAILURE);
    }
    matvar_t *names_matvar = Mat_VarRead(matfp, "names");
    if (names_matvar == NULL) {
        cerr << "Error reading names variable" << endl;
        exit(EXIT_FAILURE);
    }
    matvar_t *name_cell;
    matvar_t *potentials_matvar;
    char *function_name;
    for (int function_id = 0; function_id < NUMBER_OF_FUNCTIONS; function_id++) {
        
        initialize_psi(psi, L, num_cells);
        name_cell = Mat_VarGetCell(names_matvar, function_id);
        function_name = (char*) name_cell->data;
        
        printf("Function %d of %d: %s\n", function_id+1, NUMBER_OF_FUNCTIONS, function_name);
        
        potentials_matvar = Mat_VarRead(matfp, "potentials");
        if (potentials_matvar == NULL) {
            cerr << "Failed to read names or potentials" << endl;
            exit(EXIT_FAILURE);
        }
        load_potential(V, num_cells, function_id, (double*) potentials_matvar->data);
        pseudospec(V, L, num_cells, psi, dt, T, function_name);

    }

    /* Free Mat Variables */
    Mat_VarFree(potentials_matvar);
    Mat_VarFree(names_matvar);
    Mat_Close(matfp);
    
    
    return 0;
}