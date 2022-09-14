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

using namespace std;
using namespace std::filesystem;
using namespace Eigen;

const int RANK = 2;
const int CAPTURE_FRAME_EVERY = 500;
const double GAMMA = 0.01;
const string HOME_DIR = getenv("HOME");
const string DATA_DIR = path(HOME_DIR).append("QHD_DATA");
const string NONCVX_DIR = path(DATA_DIR).append("NonCVX-2d");
const int NUMBER_OF_FUNCTIONS = 22;

double t_dep_1(const double &t) {
    return 1 / (1 + GAMMA * t * t);
}
double t_dep_2(const double &t) {
    return 1 + GAMMA * t * t;
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

void pseudospec(const int &num_qubits, const ArrayXXd &V, const double &L, const int &N, ArrayXXcd &psi,
                const double &dt, const double &T, const string function_name)
{
    /*
    *  Pseudospectral solver for time-dependent Schrodinger equation
    *    H(t) = -0.5 * tdep(t) * Laplacian + V(x)
    *  defined on [-L,L]^2
    */
    string function_dir = path(NONCVX_DIR).append(function_name);
    string wfn_dir = path(function_dir).append(function_name + "_QHD_WFN");
    if (!filesystem::exists(function_dir)) {
        filesystem::create_directory(function_dir);
    }
    if (!filesystem::exists(wfn_dir)) {
        filesystem::create_directory(wfn_dir);
    }
    // Save wavefunction at time 0
    write_complex_to_npy(psi, path(wfn_dir).append("psi_0.npy"));

    int num_steps = T / dt;
    double t = 0;
    ArrayXd expected_potential(num_steps);
    ArrayXXd kinetic_operator(N, N);
    ArrayXXcd u(N, N);
    ArrayXXcd psi_new(N, N);
    ArrayXXcd temp(N, N);
    int i, j;
    initialize_kinetic_operator(kinetic_operator, L, N);

    int n[RANK];
    for (i = 0; i < RANK; i++)
    {
        n[i] = N;
    }

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
    // Used for computing fft2(u1) in matlab code
    in = (fftw_complex *)&u(0, 0);
    out = (fftw_complex *)&u(0, 0);
    p1 = fftw_plan_dft(RANK, n, in, out, FFTW_FORWARD, FFTW_MEASURE);

    // Used for computing ifft2(...) in matlab code
    in = (fftw_complex *)&u(0, 0);
    out = (fftw_complex *)&psi_new(0, 0);
    p2 = fftw_plan_dft(RANK, n, in, out, FFTW_BACKWARD, FFTW_MEASURE);

    double time1 = 0, time2 = 0;
    double start, end;
    start = omp_get_wtime();
    for (int step = 0; step < num_steps; step++) {

        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                temp(i,j) = exp(- I * dt * t_dep_2(t) * V(i,j));
            }
        }
        hadamard_product(temp, psi, u, N);

        fftw_execute(p1);

        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                temp(i,j) = exp(- I * dt * t_dep_1(t) * kinetic_operator(i,j));
            }
        }
        hadamard_product(temp, u, u, N);

        // ifft2
        fftw_execute(p2);

        // Normalize
        double k = pow(N, RANK);
        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                psi(i,j) = psi_new(i,j) / k;
            }
        }

        t += dt;

        if (step > 0 && step % CAPTURE_FRAME_EVERY == 0) {
            string str_t = to_string((int) (t * 10));
            string filename = "psi_" + str_t + "e-01.npy";
            write_complex_to_npy(psi, path(wfn_dir).append(filename));
        }

        // Compute expected potential
        double expected_value = 0;
        #pragma omp parallel for private(i,j) reduction(+: expected_value)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                expected_value += (psi(i,j) * conj(psi(i,j))).real() * V(i,j);
            }
        }
        expected_potential(step) = expected_value;

        printf("\r%d / %d", step + 1, num_steps);
        fflush(stdout);
    }
    end = omp_get_wtime();
    printf("\n");
    printf("Pseudospectral integrator runtime = %.5f s\n", end - start);
    
    string expected_potential_filename = "expected_potential_" + function_name + ".npy";
    write_real_to_npy_1d(expected_potential, path(function_dir).append(expected_potential_filename));
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_cleanup_threads();
}

int main(int argc, char **argv)
{

    if (argc != 6) {
        perror("Expected arguments: ./2d-pseudospec <L> <num_qubits> <T> <dt> <potentials_filename>");
        exit(EXIT_FAILURE);
    }
    const double L = stod(argv[1]);
    const int num_qubits = stoi(argv[2]);
    printf("L=%f, num_qubits=%d\n", L, num_qubits);
    const double T = stod(argv[3]);
    printf("T=%f\n", T);
    const double dt = stod(argv[4]);
    printf("dt=%f\n", dt);
    
    const char* potentials_filename;

    if (argc == 6) {
        potentials_filename = argv[5];
        cout << "Potentials filename: " << potentials_filename << endl;
    }

    // number of qubits for each dimension
    int N = pow(2, num_qubits / 2);
    printf("N=%d\n", N);

    double stepsize = 2 * L / N;
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
    ArrayXXcd psi(N, N);

    // potential V
    ArrayXXd V(N, N);

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
        
        initialize_psi(psi, L, N);
        name_cell = Mat_VarGetCell(names_matvar, function_id);
        function_name = (char*) name_cell->data;
        
        printf("Function %d of %d: %s\n", function_id+1, NUMBER_OF_FUNCTIONS, function_name);
        
        potentials_matvar = Mat_VarRead(matfp, "potentials");
        if (potentials_matvar == NULL) {
            cerr << "Failed to read names or potentials" << endl;
            exit(EXIT_FAILURE);
        }
        load_potential(V, N, function_id, (double*) potentials_matvar->data);
        pseudospec(num_qubits, V, L, N, psi, dt, T, function_name);

    }

    /* Free Mat Variables */
    Mat_VarFree(potentials_matvar);
    Mat_VarFree(names_matvar);
    Mat_Close(matfp);
    
    
    return 0;
}