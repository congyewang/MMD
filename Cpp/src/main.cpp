#include <iostream>
#include <Eigen/Dense>

typedef Eigen::Matrix<Eigen::half, 1, Eigen::Dynamic> RowVectorXhalf;
typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, 1> VectorXhalf;
typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic> MatrixXh;

float rbfKernel(const RowVectorXhalf& x, const RowVectorXhalf& y, float sigma) {
    float distanceSquared = (x.cast<float>() - y.cast<float>()).squaredNorm();
    return std::exp(-distanceSquared / (2.0 * sigma * sigma));
}

Eigen::half computeMMD(const MatrixXh& samplesP, const MatrixXh& samplesQ, float sigma) {
    int m = samplesP.rows();
    int n = samplesQ.rows();

    Eigen::half term1 = Eigen::half(0.0f);
    Eigen::half term2 = Eigen::half(0.0f);
    Eigen::half term3 = Eigen::half(0.0f);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            term1 += Eigen::half(rbfKernel(samplesP.row(i), samplesP.row(j), sigma));
        }
    }
    term1 /= Eigen::half(m * m);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            term2 += Eigen::half(rbfKernel(samplesP.row(i), samplesQ.row(j), sigma));
        }
    }
    term2 /= Eigen::half(m * n);
    term2 *= Eigen::half(2.0f);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            term3 += Eigen::half(rbfKernel(samplesQ.row(i), samplesQ.row(j), sigma));
        }
    }
    term3 /= Eigen::half(n * n);
    
    return term1 - term2 + term3;
}

int main() {
    int numSamplesP = 10000;
    int numSamplesQ = 10000;
    int dim = 2;

    VectorXhalf meanP(dim);
    MatrixXh covP(dim, dim);
    meanP << Eigen::half(0.0), Eigen::half(0.0);
    covP << Eigen::half(1.0), Eigen::half(0.0),
            Eigen::half(0.0), Eigen::half(1.0);

    VectorXhalf meanQ(dim);
    MatrixXh covQ(dim, dim);
    meanQ << Eigen::half(2.0), Eigen::half(2.0);
    covQ << Eigen::half(1.0), Eigen::half(0.0),
            Eigen::half(0.0), Eigen::half(1.0);

    Eigen::MatrixXf samplesP_float = Eigen::MatrixXf::Random(numSamplesP, dim) * covP.cast<float>().llt().matrixU() + meanP.cast<float>().transpose().replicate(numSamplesP, 1);
    Eigen::MatrixXf samplesQ_float = Eigen::MatrixXf::Random(numSamplesQ, dim) * covQ.cast<float>().llt().matrixU() + meanQ.cast<float>().transpose().replicate(numSamplesQ, 1);

    MatrixXh samplesP = samplesP_float.cast<Eigen::half>();
    MatrixXh samplesQ = samplesQ_float.cast<Eigen::half>();

    float sigma = 1.0f;
    Eigen::half mmdValue = computeMMD(samplesP, samplesQ, sigma);
    std::cout << "MMD value: " << static_cast<float>(mmdValue) << std::endl;
    return 0;
}
