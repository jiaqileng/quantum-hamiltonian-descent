#include <string>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
void write_real_to_npy_1d(const ArrayXd &array, string filename);
void write_real_to_npy_2d(const ArrayXXd &array, string filename);
void write_complex_to_npy(const ArrayXXcd &array, string filename);