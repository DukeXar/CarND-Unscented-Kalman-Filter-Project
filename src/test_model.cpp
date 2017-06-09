#include "model.h"

#include <gtest/gtest.h>
#include "Eigen/Dense"

namespace {

::testing::AssertionResult MatricesAlmostEqual(const Eigen::MatrixXd& m1,
                                               const Eigen::MatrixXd& m2) {
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
  Eigen::VectorXd x(n_x);
  x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

  Eigen::MatrixXd cov(n_x, n_x);
  // clang-format off
  cov <<
     0.0043, -0.0013,  0.0030, -0.0022, -0.0020,
    -0.0013,  0.0077,  0.0011,  0.0071,  0.0060,
     0.0030,  0.0011,  0.0054,  0.0007,  0.0008,
    -0.0022,  0.0071,  0.0007,  0.0098,  0.0100,
    -0.0020,  0.0060,  0.0008,  0.0100,  0.0123;
  // clang-format on

  Eigen::MatrixXd expected(n_x, 2 * n_x + 1);
  // clang-format off
  expected <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879;
  // clang-format on

  Eigen::MatrixXd gen =
      Model::Details::GenerateSigmaPoints(Model::Gaussian{x, cov});

  ASSERT_TRUE(MatricesAlmostEqual(gen, expected));
}

TEST(TestGenerateAugmentedSigmaPoints, SmokeTest) {
  const size_t n_x = 5;
  Eigen::VectorXd x(n_x);
  x << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;

  Eigen::MatrixXd cov(n_x, n_x);
  // clang-format off
  cov <<
     0.0043, -0.0013,  0.0030, -0.0022, -0.0020,
    -0.0013,  0.0077,  0.0011,  0.0071,  0.0060,
     0.0030,  0.0011,  0.0054,  0.0007,  0.0008,
    -0.0022,  0.0071,  0.0007,  0.0098,  0.0100,
    -0.0020,  0.0060,  0.0008,  0.0100,  0.0123;
  // clang-format on

  // clang-format off
  Eigen::MatrixXd expected(n_x + 2, (n_x+2) * 2 + 1);
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

  Eigen::MatrixXd aug = Model::Details::CTRVGenerateAugmentedSigmaPoints(
      Model::Gaussian{x, cov}, std_a, std_yawdd);

  EXPECT_TRUE(MatricesAlmostEqual(aug, expected));
}

TEST(TestPredictSigmaPoints, SmokeTest) {
  const size_t n_x = 5;
  const size_t n_aug = 7;

  // clang-format off
  Eigen::MatrixXd sig_aug(n_aug, 2 * n_aug + 1);
  sig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
  // clang-format on

  Eigen::MatrixXd expected(n_x, 2 * n_aug + 1);
  // clang-format off
  expected <<
    5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516, 5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589, 5.93401,  5.93553,
    1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001, 1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876, 1.48855,  1.48939,
     2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049, 2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049, 2.17026,   2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
     0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,  0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,  0.3528, 0.318159;
  // clang-format on

  const double dt = 0.1;

  Eigen::MatrixXd pred = Model::Details::CTRVPredictSigmaPoints(sig_aug, dt);
  EXPECT_TRUE(MatricesAlmostEqual(pred, expected)) << "pred=\n" << pred;
}

TEST(TestSigmaPointsToGaussian, SmokeTest) {
  const size_t n_x = 5;
  const size_t n_aug = 7;

  Eigen::MatrixXd sig_pred(n_x, 2 * n_aug + 1);
  // clang-format off
  sig_pred <<
    5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
      1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
     2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
    0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
     0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
  // clang-format on

  Eigen::VectorXd expected_x(n_x);
  expected_x << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  Eigen::MatrixXd expected_cov(n_x, n_x);
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

  Eigen::MatrixXd sig_pred(n_x, 2 * n_aug + 1);
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

  Eigen::MatrixXd radar_cov(radar_n_x, radar_n_x);
  // clang-format off
  radar_cov <<
    std_radr*std_radr,                        0, 0,
                    0, std_radphi*std_radphi, 0,
                    0,                        0, std_radrd*std_radrd;
  // clang-format on

  Eigen::VectorXd expected_mean(radar_n_x);
  expected_mean << 6.12155, 0.245993, 2.10313;

  Eigen::MatrixXd expected_cov(radar_n_x, radar_n_x);
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

}  // namespace