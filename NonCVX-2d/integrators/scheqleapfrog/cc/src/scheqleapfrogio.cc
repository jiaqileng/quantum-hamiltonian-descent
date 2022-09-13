#include <complex>
#include <fstream>
#include <iostream>
#include <string>

#include <eigen3/Eigen/Eigen>

using namespace std::complex_literals;


void read_initial_state(const char* input_fname, Eigen::VectorXcd& psi)
{
    std::ifstream init_state_fstream(input_fname);

    std::string re_str;
    std::string im_str;

    double re;
    double im;

    std::vector<std::complex<double>> psi_vec;

    while (init_state_fstream >> re_str >> im_str)
    {
        re = std::stod(re_str);
        im = std::stod(im_str);

        psi_vec.push_back(re + 1i * im);
    }

    psi.resize(psi_vec.size());

    for (int elem_idx = 0; elem_idx < psi_vec.size(); ++elem_idx)
    {
        psi(elem_idx) = psi_vec[elem_idx];
    }

    init_state_fstream.close();
}


void read_laplacian(const char* input_fname, Eigen::SparseMatrix<double, Eigen::RowMajor>& laplacian)
{
    std::ifstream laplacian_fstream(input_fname);

    std::string row_str;
    std::string col_str;
    std::string data_str;

    int row;
    int col;
    double data;

    std::vector<Eigen::Triplet<double>> triplets;

    int count_diag = 0;

    while (laplacian_fstream >> row_str >> col_str >> data_str)
    {
        row = std::stoi(row_str);
        col = std::stoi(col_str);
        data = std::stod(data_str);

        if (row == col)
        {
            ++count_diag;
        }

        triplets.push_back(Eigen::Triplet<double>(row, col, data));
    }

    laplacian.resize(count_diag, count_diag);
    laplacian.reserve(triplets.size());

    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    laplacian.makeCompressed();
}


void write_state_snapshots(std::vector<Eigen::VectorXcd>& snapshots, const char* output_fname)
{
    Eigen::IOFormat vec_to_row_fmt(Eigen::FullPrecision,
        0,
        " ",
        " ",
        "",
        "",
        "",
        ""
    );

    std::ofstream state_snapshot_fstream(output_fname);

    for (Eigen::VectorXcd& snapshot : snapshots)
    {
        state_snapshot_fstream << snapshot.format(vec_to_row_fmt) << std::endl;
    }

    state_snapshot_fstream.close();
}
