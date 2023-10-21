#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

float rbfKernel(const RowVectorXf& x, const RowVectorXf& y, float sigma) {
    float distanceSquared = (x - y).squaredNorm();
    return std::exp(-distanceSquared / (2.0 * sigma * sigma));
}

float computeMMD(const MatrixXf& samplesP, const MatrixXf& samplesQ, float sigma) {
    int m = samplesP.rows();
    int n = samplesQ.rows();

    float term1 = 0.0f;
    float term2 = 0.0f;
    float term3 = 0.0f;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            term1 += float(rbfKernel(samplesP.row(i), samplesP.row(j), sigma));
        }
    }
    term1 /= float(m * m);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            term2 += float(rbfKernel(samplesP.row(i), samplesQ.row(j), sigma));
        }
    }
    term2 /= float(m * n);
    term2 *= float(2.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            term3 += float(rbfKernel(samplesQ.row(i), samplesQ.row(j), sigma));
        }
    }
    term3 /= float(n * n);
    
    return term1 - term2 + term3;
}

int main() {
    // Example usage
    MatrixXf X(3, 3);
    X << 1, 2, 3,
          4, 5, 6,
          7, 8, 9;

    MatrixXf Y(4, 3);
    Y << 7, 6, 5,
          4, 3, 2,
          1, 1, 8,
          0, 2, 5;

    float sigma = 1.0f;
    float mmdValue = computeMMD(X, Y, sigma);

    std::cout << "MMD value: " << mmdValue << std::endl;

    return 0;
}
