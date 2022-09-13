#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
//#include <endian.h>
using namespace Eigen;
using namespace std;

const int MAGIC_LEN = 8;
const int ARRAY_ALIGN = 64;

void write_real_to_npy_1d(const ArrayXd &array, string filename) {

    if (array.size() == 0) {
        perror("array is empty");
        exit(EXIT_FAILURE);
    }

    ofstream file;
    file.open(filename, ios::binary);
    file.write("\x93NUMPY", 6);
    file.write("\x01", 1);
    file.write("\x00", 1);
    string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + to_string(array.size()) + ", " + "), }";
    unsigned short hlen = header.length() + 1;
    int padlen = ARRAY_ALIGN - ((MAGIC_LEN + 2 + hlen) % ARRAY_ALIGN);

    unsigned short len = hlen + padlen;
    //len = htole64(len);
    file.write(reinterpret_cast<const char *>(&len), sizeof(len));

    for (char c : header) {
        file.write((char *) &c, 1);
    }
    for (int i = 0; i < padlen; i++) {
        file.write(" ", 1);
    }
    file.write("\n", 1);

    for (int i = 0; i < array.rows(); i++) {
        file.write(reinterpret_cast<const char *>(&array(i)), sizeof(double));
    }
    file.close();
}

// Writes a 2D array of doubles into numpy array (float64)
void write_real_to_npy_2d(const ArrayXXd &array, string filename) {

    if (array.size() == 0) {
        perror("array is empty");
        exit(EXIT_FAILURE);
    }

    ofstream file;
    file.open(filename, ios::binary);
    file.write("\x93NUMPY", 6);
    file.write("\x01", 1);
    file.write("\x00", 1);
    string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + to_string(array.rows()) + ", " + to_string(array.cols()) + "), }";
    unsigned short hlen = header.length() + 1;
    int padlen = ARRAY_ALIGN - ((MAGIC_LEN + 2 + hlen) % ARRAY_ALIGN);

    unsigned short len = hlen + padlen;
    //len = htole64(len);
    file.write(reinterpret_cast<const char *>(&len), sizeof(len));

    for (char c : header) {
        file.write((char *) &c, 1);
    }
    for (int i = 0; i < padlen; i++) {
        file.write(" ", 1);
    }
    file.write("\n", 1);

    for (int i = 0; i < array.rows(); i++) {
        for (int j = 0; j < array.cols(); j++) {
            file.write(reinterpret_cast<const char *>(&array(i,j)), sizeof(double));
        }
    }
    file.close();
}

// Writes a 2D array of doubles into numpy array (float64)
void write_complex_to_npy(const ArrayXXcd &array, string filename) {

    if (array.size() == 0) {
        perror("array is empty");
        exit(EXIT_FAILURE);
    }

    ofstream file;
    file.open(filename, ios::binary);
    file.write("\x93NUMPY", 6);
    file.write("\x01", 1);
    file.write("\x00", 1);
    string header = "{'descr': '<c16', 'fortran_order': False, 'shape': (" + to_string(array.rows()) + ", " + to_string(array.cols()) + "), }";
    unsigned short hlen = header.length() + 1;
    int padlen = ARRAY_ALIGN - ((MAGIC_LEN + 2 + hlen) % ARRAY_ALIGN);

    unsigned short len = hlen + padlen;
    //len = htole64(len);
    file.write(reinterpret_cast<const char *>(&len), sizeof(len));

    for (char c : header) {
        file.write((char *) &c, 1);
    }
    for (int i = 0; i < padlen; i++) {
        file.write(" ", 1);
    }
    file.write("\n", 1);

    ArrayXXd real = array.real();
    ArrayXXd imag = array.imag();
    for (int i = 0; i < array.rows(); i++) {
        for (int j = 0; j < array.cols(); j++) {
            file.write(reinterpret_cast<const char *>(&real(i,j)), sizeof(double));
            file.write(reinterpret_cast<const char *>(&imag(i,j)), sizeof(double));
        }
    }
    file.close();
}
