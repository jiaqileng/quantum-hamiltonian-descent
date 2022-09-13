#include <iostream>
#include <complex>
#include <math.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include "to_npy.hpp"
#include <omp.h>
#include <matio.h>
#include <experimental/filesystem>

#define I complex<double>(0, 1)

using namespace std;
using namespace Eigen;
const int RANK = 2;
const int CAPTURE_FRAME_EVERY = 500;
const string DATA_DIR = "data2/";
const string EXPECTED_POTENTIALS_DIR = "expected_potentials2/";
const string POTENTIALS_DIR = "potentials2/";
const int NUMBER_OF_FUNCTIONS = 22;
/*double sqrt_gaussian(const double &x1, const double &x2, const double &stepsize) {
    double sigma = 1;
    double x1_mean = 3;
    double x2_mean = 2;
    //return (stepsize / (sigma * sqrt(2 * M_PI))) * exp(-((x1 - x1_mean) * (x1 - x1_mean) + (x2 - x2_mean) * (x2 - x2_mean)) / (4 * sigma * sigma));
    return exp(-((x1 - x1_mean) * (x1 - x1_mean) + (x2 - x2_mean) * (x2 - x2_mean)) / (4 * sigma * sigma));
}*/
double potential_function(const double &x1, const double &x2) {
    return 5 * (x1 * x1 + x2 * x2);
}
double t_dep_1(const double &t) {
    return 1 / (1 + 0.01 * t * t);
}
double t_dep_2(const double &t) {
    return 1 + 0.01 * t * t;
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

    // l2 normalization
    /*double sqrt_sum = sqrt(sum);
    #pragma omp parallel for private(i,j) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            y(i, j) /= sqrt_sum;
        }
    }*/

    // verify
    sum = 0;
    #pragma omp parallel for private(i,j) reduction(+: sum) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum += (y(i, j) * conj(y(i, j))).real();
        }
    }
    printf("sum=%f\n", sum);
}
void initialize_potential(ArrayXXd &V, const double &L, const int &N) {
    double x1, x2;
    int i, j;
    double stepsize = 2 * L / N;

    #pragma omp parallel for private(i, j, x1, x2) schedule(static)
    for (i = 0; i < N; i++) {
        x1 = -L + i * stepsize;
        for (j = 0; j < N; j++) {
            x2 = -L + j * stepsize;
            // transpose to be consistent with matlab
            V(j, i) = potential_function(x1, x2);
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

// debugging purposes
/*void get_prob_sum(const ArrayXXcd &psi0) {
    int n = psi0.rows();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += (psi0(i,j) * conj(psi0(i,j))).real();
        }
    }
    printf("sum=%f\n", sum);
}*/
void hadamard_product(ArrayXXcd &A, ArrayXXcd &B, ArrayXXcd &out, const int &N) {
    int i, j;
    #pragma omp parallel for private(i,j) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            out(i,j) = A(i,j) * B(i,j);
        }
    }
}

void pseudospec(const int &num_qubits, const ArrayXXd &V, const double &L, const int &N, ArrayXXcd &psi0,
                const double &dt, const double &T, const string function_name)
{
    /*
    *  Pseudospectral solver for time-dependent Schrodinger equation
    *    H(t) = -0.5 * tdep(t) * Laplacian + V(x)
    *  defined on [-L,L]^2
    */

    int num_steps = T / dt;
    double t = 0;
    ArrayXd expected_potential(num_steps);
    // part 2
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
        //u = exp(-I * dt * t_dep_2(t) * V) * psi0;
        
        #pragma omp parallel for private(i,j) schedule(static)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                temp(i,j) = exp(- I * dt * t_dep_2(t) * V(i,j));
            }
        }
        hadamard_product(temp, psi0, u, N);
        // fft2(u1)
        fftw_execute(p1);
        // exp(-1i.*dt.*coeffT.*kinetic) .* fft2(u1)
        //u = exp(- I * dt * t_dep_1(t) * kinetic_operator) * u;

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
                psi0(i,j) = psi_new(i,j) / k;
            }
        }

        if (step % CAPTURE_FRAME_EVERY == 0) {
            string str_t = to_string(t);
            string padded_t = string(10 - str_t.length(), '0') + str_t;
            write_complex_to_npy(psi0, DATA_DIR + function_name + "/" + "psi_" + padded_t + ".npy");
        }
        // Compute expected potential
        double expected_value = 0;
        #pragma omp parallel for private(i,j) reduction(+: expected_value)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                expected_value += (psi0(i,j) * conj(psi0(i,j))).real() * V(i,j);
            }
        }
        expected_potential(step) = expected_value;
        //expected_potential(step) = ((psi0 * psi0.conjugate()).real() * V).sum();
        //printf("%f\n", expected_potential(step));

        printf("\r%d / %d", step + 1, num_steps);
        fflush(stdout);
        t += dt;
    }
    end = omp_get_wtime();
    printf("\n");
    printf("Pseudospectral integrator runtime = %.5f s\n", end - start);
    write_real_to_npy_1d(expected_potential, EXPECTED_POTENTIALS_DIR + "expected_potential_" + function_name + ".npy");
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_cleanup_threads();
}

int main(int argc, char **argv)
{

    if (argc != 5 && argc != 6) {
        perror("Expected arguments: ./2d-pseudospec <L> <num_qubits> <T> <dt> [<potentials_filename>]");
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
        cout << potentials_filename << endl;
    }

    // num_qubits can store integers from [-2^num_qubits, 2^num_qubits - 1]
    int N = pow(2, num_qubits / 2);
    printf("N=%d\n", N);

    double stepsize = 2 * L / N;
    printf("stepsize=%f\n", stepsize);

    printf("max threads: %d\n", omp_get_num_procs());
    printf("threads: %d\n", omp_get_max_threads());

    if (!experimental::filesystem::is_directory(DATA_DIR)) {
        experimental::filesystem::create_directory(DATA_DIR);
    }
    if (!experimental::filesystem::is_directory(POTENTIALS_DIR)) {
        experimental::filesystem::create_directory(POTENTIALS_DIR);
    }
    if (!experimental::filesystem::is_directory(EXPECTED_POTENTIALS_DIR)) {
        experimental::filesystem::create_directory(EXPECTED_POTENTIALS_DIR);
    }
    
    // psi_0
    ArrayXXcd psi_0(N, N);

    // potential V
    ArrayXXd V(N, N);

    if (argc == 5) {
        if (!experimental::filesystem::is_directory(DATA_DIR + "test" + "/")) {
            experimental::filesystem::create_directory(DATA_DIR + "test" + "/");
        }
        initialize_psi(psi_0, L, N);
        initialize_potential(V, L, N);
        write_real_to_npy_2d(V, POTENTIALS_DIR + "potential_test" + ".npy");
        pseudospec(num_qubits, V, L, N, psi_0, dt, T, "test");
    } else {
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
            
            initialize_psi(psi_0, L, N);
            name_cell = Mat_VarGetCell(names_matvar, function_id);
            function_name = (char*) name_cell->data;

            if (!experimental::filesystem::is_directory(DATA_DIR + function_name + "/")) {
                experimental::filesystem::create_directory(DATA_DIR + function_name + "/");
            }
            
            printf("Function %d of %d: %s\n", function_id+1, NUMBER_OF_FUNCTIONS, function_name);
            
            potentials_matvar = Mat_VarRead(matfp, "potentials");
            if (potentials_matvar == NULL) {
                cerr << "Failed to read names or potentials" << endl;
                exit(EXIT_FAILURE);
            }
            load_potential(V, N, function_id, (double*) potentials_matvar->data);
            if (!experimental::filesystem::is_directory(POTENTIALS_DIR)) {
                experimental::filesystem::create_directory(POTENTIALS_DIR);
            }
            write_real_to_npy_2d(V, POTENTIALS_DIR + "potential_" + function_name + ".npy");
            
            pseudospec(num_qubits, V, L, N, psi_0, dt, T, function_name);

        }

        /* Free Mat Variables */
        Mat_VarFree(potentials_matvar);
        Mat_VarFree(names_matvar);
        Mat_Close(matfp);
    }
    
    return 0;
}