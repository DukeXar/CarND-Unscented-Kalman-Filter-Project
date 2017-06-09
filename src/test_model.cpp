#include "model.h"

#include <gtest/gtest.h>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

::testing::AssertionResult MatricesAlmostEqual(const MatrixXd& m1,
                                               const MatrixXd& m2) {
  if (m1.cols() != m2.cols()) {
    return ::testing::AssertionFailure()
           << "number of colums differs (" << m1.cols() << ", " << m2.cols()
           << ")";
  }

  if (m1.rows() != m2.rows()) {
    return ::testing::AssertionFailure()
           << "number of rows differs (" << m1.rows() << ", " << m2.rows()
           << ")";
  }

  for (size_t r = 0; r < m1.rows(); ++r) {
    for (size_t c = 0; c < m1.cols(); ++c) {
      const double threshold = 0.00001;

      const double delta = m1(r, c) - m2(r, c);
      if (fabs(delta) > threshold) {
        return ::testing::AssertionFailure()
               << "m1(" << r << "," << c << ") <> m2(" << r << "," << c
               << "), where m1(" << r << "," << c << ") = " << m1(r, c)
               << ", m2(" << r << "," << c << ") = " << m2(r, c);
      }
    }
  }

  return ::testing::AssertionSuccess();
}

TEST(TestGenerateSigmaPoints, SanityCheck) {
  const size_t n_x = 5;
  VectorXd x(n_x);
  x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

  MatrixXd cov(n_x, n_x);
  // clang-format off
  cov <<
     0.0043, -0.0013,  0.0030, -0.0022, -0.0020,
    -0.0013,  0.0077,  0.0011,  0.0071,  0.0060,
     0.0030,  0.0011,  0.0054,  0.0007,  0.0008,
    -0.0022,  0.0071,  0.0007,  0.0098,  0.0100,
    -0.0020,  0.0060,  0.0008,  0.0100,  0.0123;
  // clang-format on

  MatrixXd expected(n_x, 2 * n_x + 1);
  // clang-format off
  expected <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879;
  // clang-format on

  MatrixXd gen = Model::Details::GenerateSigmaPoints(Model::Gaussian{x, cov});

  ASSERT_TRUE(MatricesAlmostEqual(gen, expected));
}

TEST(TestGenerateAugmentedSigmaPoints, SmokeTest) {
  const size_t n_x = 5;
  VectorXd x(n_x);
  x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

  MatrixXd cov(n_x, n_x);
  // clang-format off
  cov <<
     0.0043, -0.0013,  0.0030, -0.0022, -0.0020,
    -0.0013,  0.0077,  0.0011,  0.0071,  0.0060,
     0.0030,  0.0011,  0.0054,  0.0007,  0.0008,
    -0.0022,  0.0071,  0.0007,  0.0098,  0.0100,
    -0.0020,  0.0060,  0.0008,  0.0100,  0.0123;
  // clang-format on

  // clang-format off
  MatrixXd expected(n_x + 2, (n_x+2) * 2 + 1);
  expected <<
  5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
    1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
  2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
  0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
  0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
       0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
       0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;
  // clang-format on

  const double std_a = 0.2;
  const double std_yawdd = 0.2;

  MatrixXd aug = Model::Details::CTRVGenerateAugmentedSigmaPoints(
      Model::Gaussian{x, cov}, std_a, std_yawdd);

  EXPECT_TRUE(MatricesAlmostEqual(aug, expected));
}

TEST(TestPredictSigmaPoints, SmokeTest) {
  const size_t n_x = 5;
  const size_t n_aug = 7;

  // clang-format off
  MatrixXd sig_aug(n_aug, 2 * n_aug + 1);
  sig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
  // clang-format on

  MatrixXd expected(n_x, 2 * n_aug + 1);
  // clang-format off
  expected <<
    5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516, 5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589, 5.93401,  5.93553,
    1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001, 1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876, 1.48855,  1.48939,
     2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049, 2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049, 2.17026,   2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
     0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,  0.3528, 0.318159;
  // clang-format on

  const double dt = 0.1;

  MatrixXd pred = Model::Details::CTRVPredictSigmaPoints(sig_aug, dt);
  EXPECT_TRUE(MatricesAlmostEqual(pred, expected)) << "pred=\n" << pred;
}

TEST(TestSigmaPointsToGaussian, SmokeTest) {
  const size_t n_x = 5;
  const size_t n_aug = 7;

  MatrixXd sig_pred(n_x, 2 * n_aug + 1);
  // clang-format off
  sig_pred <<
    5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
      1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
     2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
    0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
     0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
  // clang-format on

  VectorXd expected_x(n_x);
  expected_x << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  MatrixXd expected_cov(n_x, n_x);
  // clang-format off
  expected_cov <<
    0.00543425, -0.0024053,  0.00341576, -0.00348196, -0.00299378,
    -0.0024053,   0.010845,   0.0014923,  0.00980182,  0.00791091,
    0.00341576,  0.0014923,  0.00580129, 0.000778632, 0.000792973,
   -0.00348196, 0.00980182, 0.000778632,   0.0119238,   0.0112491,
   -0.00299378, 0.00791091, 0.000792973,   0.0112491,   0.0126972;
  // clang-format on

  auto gauss = Model::Details::SigmaPointsToGaussian(
      sig_pred, Model::Details::CTRVNormalizeStateAngle);

  EXPECT_TRUE(MatricesAlmostEqual(gauss.mean, expected_x)) << "mean=\n"
                                                           << gauss.mean;
  EXPECT_TRUE(MatricesAlmostEqual(gauss.cov, expected_cov)) << "cov=\n"
                                                            << gauss.cov;
}

TEST(TestPredictRadarMeasurement, SmokeTest) {
  const size_t n_x = 5;
  const size_t n_aug = 7;
  const size_t radar_n_x = 3;

  MatrixXd sig_pred(n_x, 2 * n_aug + 1);
  // clang-format off
  sig_pred <<
    5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
      1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
     2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
    0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
     0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
  // clang-format on

  const double std_radr = 0.3;
  const double std_radphi = 0.0175;
  const double std_radrd = 0.1;

  MatrixXd radar_cov(radar_n_x, radar_n_x);
  // clang-format off
  radar_cov <<
    std_radr*std_radr,                        0, 0,
                    0, std_radphi*std_radphi, 0,
                    0,                        0, std_radrd*std_radrd;
  // clang-format on

  VectorXd expected_mean(radar_n_x);
  expected_mean << 6.12155, 0.245993, 2.10313;

  MatrixXd expected_cov(radar_n_x, radar_n_x);
  // clang-format off
  expected_cov <<
      0.0946171, -0.000139448,   0.00407016,
   -0.000139448,  0.000617548, -0.000770652,
     0.00407016, -0.000770652,    0.0180917;
  // clang-format on

  // This is what CTRVUnscentedKalmanFilter::Update does before KalmanUpdate
  // call
  using namespace Model::Details;
  auto pred_measurement_sigma_pts = ApplySensorModel<Model::Radar>(sig_pred);
  auto pred_measurement = SigmaPointsToGaussian(pred_measurement_sigma_pts,
                                                &Model::Radar::NormalizeDelta);
  pred_measurement.cov += radar_cov;

  EXPECT_TRUE(MatricesAlmostEqual(pred_measurement.mean, expected_mean))
      << "mean=\n"
      << pred_measurement.mean;
  EXPECT_TRUE(MatricesAlmostEqual(pred_measurement.cov, expected_cov))
      << "cov=\n"
      << pred_measurement.cov;
}

TEST(TestUpdateState, RadarUpdate) {
  const size_t n_x = 5;
  const size_t n_aug = 7;
  const size_t n_z = 3;

  // TODO(dukexar): Naming, this was just copy-paste from the lesson.

  // create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  // clang-format off
  Xsig_pred <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
  // clang-format on

  // create example vector for predicted state mean
  VectorXd x = VectorXd(n_x);
  x << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  // create example matrix for predicted state covariance
  MatrixXd P = MatrixXd(n_x, n_x);
  // clang-format off
  P <<
    0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
    -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
    0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
   -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
   -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;
  // clang-format on

  // create example matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  // clang-format off
  Zsig <<
      6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
     0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
      2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;
  // clang-format on

  // create example vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred << 6.12155, 0.245993, 2.10313;

  // create example matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);
  // clang-format off
  S <<
      0.0946171, -0.000139448,   0.00407016,
   -0.000139448,  0.000617548, -0.000770652,
     0.00407016, -0.000770652,    0.0180917;
  // clang-format on

  // create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << 5.9214, 0.2187, 2.0062;

  VectorXd expected_mean(n_x);
  expected_mean << 5.92276, 1.41823, 2.15593, 0.489274, 0.321338;

  MatrixXd expected_cov(n_x, n_x);
  // clang-format off
  expected_cov <<
      0.00361579, -0.000357881, 0.00208316, -0.000937196, -0.00071727,
    -0.000357881,   0.00539867, 0.00156846,   0.00455342,  0.00358885,
      0.00208316,   0.00156846, 0.00410651,   0.00160333,  0.00171811,
    -0.000937196,   0.00455342, 0.00160333,   0.00652634,  0.00669436,
     -0.00071719,   0.00358884, 0.00171811,   0.00669426,  0.00881797;
  // clang-format on

  Model::Gaussian result = Model::Details::KalmanUpdate(
      Model::Gaussian{x, P}, Xsig_pred, Model::Gaussian{z_pred, S}, Zsig, z);
  EXPECT_TRUE(MatricesAlmostEqual(result.mean, expected_mean)) << "mean=\n"
                                                               << result.mean;
  EXPECT_TRUE(MatricesAlmostEqual(result.cov, expected_cov)) << "cov=\n"
                                                             << result.cov;
}

}  // namespace