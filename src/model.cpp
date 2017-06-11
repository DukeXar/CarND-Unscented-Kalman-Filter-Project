#include "model.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace Model {

const size_t kNoiseCovSize = 2;
const size_t kAugStateSize = kStateSize + kNoiseCovSize;

namespace Details {
Eigen::MatrixXd GenerateSigmaPoints(const Gaussian& state) {
  const size_t n_x = state.mean.rows();
  // It is important to have 3.0 as double, otherwise compiler bites
  const double lambda = 3.0 - n_x;

  const Eigen::MatrixXd sqrt_cov = state.cov.llt().matrixL();

  Eigen::MatrixXd result(n_x, 2 * n_x + 1);
  result.col(0) = state.mean;

  for (size_t i = 0; i < n_x; ++i) {
    const auto k = sqrt(lambda + n_x) * sqrt_cov.col(i);
    result.col(i + 1) = state.mean + k;
    result.col(i + 1 + n_x) = state.mean - k;
  }

  return result;
}

Eigen::MatrixXd CTRVGenerateAugmentedSigmaPoints(const Gaussian& state,
                                                 double std_a,
                                                 double std_yawdd) {
  const size_t n_x = state.mean.rows();
  const size_t aug_n_x = n_x + kNoiseCovSize;

  Eigen::VectorXd aug_state(aug_n_x);
  aug_state.fill(0);
  aug_state.head(n_x) = state.mean;

  Eigen::MatrixXd aug_cov(aug_n_x, aug_n_x);
  aug_cov.fill(0);
  aug_cov.topLeftCorner(n_x, n_x) = state.cov;
  aug_cov(n_x, n_x) = std_a * std_a;
  aug_cov(n_x + 1, n_x + 1) = std_yawdd * std_yawdd;

  return GenerateSigmaPoints(Gaussian{aug_state, aug_cov});
}

Eigen::VectorXd CTRVPredictVector(const Eigen::VectorXd& aug_sigma_pts,
                                  double dt) {
  if (aug_sigma_pts.rows() != kAugStateSize) {
    throw std::runtime_error("Invalid input");
  }

  const double px = aug_sigma_pts(0);
  const double py = aug_sigma_pts(1);
  const double v = aug_sigma_pts(2);
  const double psi = aug_sigma_pts(3);
  const double psi_dot = aug_sigma_pts(4);
  const double nu_a = aug_sigma_pts(5);
  const double nu_psi_dotdot = aug_sigma_pts(6);

  double px_new, py_new;
  if (fabs(psi_dot) > 0.00001) {
    px_new = px + v / psi_dot * (sin(psi + psi_dot * dt) - sin(psi));
    py_new = py + v / psi_dot * (cos(psi) - cos(psi + psi_dot * dt));
  } else {
    px_new = px + v * dt * cos(psi);
    py_new = py + v * dt * sin(psi);
  }

  const double psi_new = psi + psi_dot * dt;

  const double px_new_noise = 0.5 * dt * dt * nu_a * cos(psi);
  const double py_new_noise = 0.5 * dt * dt * nu_a * sin(psi);

  const double v_new_noise = dt * nu_a;

  const double psi_new_noise = 0.5 * dt * dt * nu_psi_dotdot;
  const double psi_dot_new_noise = dt * nu_psi_dotdot;

  Eigen::VectorXd result(kStateSize);
  result(0) = px_new + px_new_noise;
  result(1) = py_new + py_new_noise;
  result(2) = v + v_new_noise;
  result(3) = psi_new + psi_new_noise;
  result(4) = psi_dot + psi_dot_new_noise;

  return result;
}

Eigen::MatrixXd CTRVPredictSigmaPoints(const Eigen::MatrixXd& aug_sigma_pts,
                                       double dt) {
  Eigen::MatrixXd result(kStateSize, aug_sigma_pts.cols());

  for (size_t i = 0; i < aug_sigma_pts.cols(); ++i) {
    result.col(i) = CTRVPredictVector(aug_sigma_pts.col(i), dt);
  }

  return result;
}

double NormalizeAngle(double angle) {
  bool is_neg = false;

  if (angle < 0) {
    angle = -angle;
    is_neg = true;
  }

  // angle >= 0.0
  // angle + M_PI >= M_PI
  // new_angle in [0.0 .. 2 * M_PI)
  // new_angle == 0.0 only if (angle + M_PI) == (K * 2 * M_PI)

  double new_angle = fmod(angle + M_PI, 2 * M_PI);
  if (new_angle == 0.0) {
    new_angle = M_PI;
  } else {
    new_angle -= M_PI;
  }

  if (is_neg) {
    return -new_angle;
  }

  return new_angle;
}

void CTRVNormalizeStateAngle(Eigen::VectorXd& state) {
  state(3) = NormalizeAngle(state(3));
}

// Generates a vector of weights for given number of sigma points
Eigen::MatrixXd GenerateSigmaWeights(size_t n_pts) {
  const size_t n_aug = (n_pts - 1) / 2;
  const double lambda = 3.0 - n_aug;

  Eigen::VectorXd weights(n_pts);
  weights(0) = lambda / (lambda + n_aug);
  for (size_t i = 1; i < weights.rows(); ++i) {
    weights(i) = 0.5 / (lambda + n_aug);
  }

  return weights;
}

Gaussian SigmaPointsToGaussian(
    const Eigen::MatrixXd& pred_sigma_pts,
    const std::function<void(Eigen::VectorXd&)>& normalizer) {
  const size_t n_sigma_pts = pred_sigma_pts.cols();
  const size_t state_size = pred_sigma_pts.rows();

  Eigen::VectorXd weights = GenerateSigmaWeights(n_sigma_pts);

  Eigen::VectorXd res_x(state_size);
  res_x.fill(0);
  for (size_t i = 0; i < n_sigma_pts; ++i) {
    res_x += weights(i) * pred_sigma_pts.col(i);
  }

  Eigen::MatrixXd res_cov(state_size, state_size);
  res_cov.fill(0);
  for (size_t i = 0; i < n_sigma_pts; ++i) {
    Eigen::VectorXd diff = pred_sigma_pts.col(i) - res_x;
    normalizer(diff);
    res_cov += weights(i) * diff * diff.transpose();
  }

  return {res_x, res_cov};
}

Eigen::MatrixXd CreateCrossCorrelation(
    const Eigen::VectorXd& state_mean, const Eigen::MatrixXd& state_sigma_pts,
    const Eigen::VectorXd& measurement_mean,
    const Eigen::MatrixXd& measurement_sigma_pts) {
  const size_t n_sigma_pts = state_sigma_pts.cols();
  Eigen::VectorXd weights = GenerateSigmaWeights(n_sigma_pts);
  Eigen::MatrixXd result(state_mean.rows(), measurement_mean.rows());
  result.fill(0);
  for (size_t i = 0; i < n_sigma_pts; ++i) {
    auto state_diff = (state_sigma_pts.col(i) - state_mean);
    auto measurement_diff = (measurement_sigma_pts.col(i) - measurement_mean);
    result += weights(i) * state_diff * measurement_diff.transpose();
  }
  return result;
}

KalmanUpdateResult KalmanUpdate(
    const Gaussian& pred_state, const Eigen::MatrixXd& pred_state_sigma_pts,
    const Gaussian& pred_measurement,
    const Eigen::MatrixXd& pred_measurement_sigma_pts,
    const Eigen::VectorXd& measurement_mean) {
  Eigen::MatrixXd cross_corr =
      CreateCrossCorrelation(pred_state.mean, pred_state_sigma_pts,
                             pred_measurement.mean, pred_measurement_sigma_pts);

  Eigen::MatrixXd s_inv = pred_measurement.cov.inverse();
  Eigen::MatrixXd kalman_gain = cross_corr * s_inv;

  auto measurement_diff = (measurement_mean - pred_measurement.mean);
  Eigen::VectorXd new_state_mean =
      pred_state.mean + kalman_gain * measurement_diff;

  Eigen::MatrixXd new_state_cov =
      pred_state.cov -
      kalman_gain * pred_measurement.cov * kalman_gain.transpose();

  double nis = measurement_diff.transpose() * s_inv * measurement_diff;
  return {Gaussian{new_state_mean, new_state_cov}, nis};
}

}  // namespace Details

CTRVUnscentedKalmanFilter::CTRVUnscentedKalmanFilter(double std_a,
                                                     double std_yawdd,
                                                     Gaussian initialState)
    : std_a_(std_a), std_yawdd_(std_yawdd), state_(initialState), nis_() {}

void CTRVUnscentedKalmanFilter::Predict(double dt) {
  using namespace Details;
  sigma_pts_ = CTRVPredictSigmaPoints(
      CTRVGenerateAugmentedSigmaPoints(state_, std_a_, std_yawdd_), dt);
  state_ = SigmaPointsToGaussian(sigma_pts_, CTRVNormalizeStateAngle);
}

Gaussian Radar::CreateInitialGaussian(const Eigen::VectorXd& measurement,
                                      const Eigen::MatrixXd& measurement_cov) {
  const double rho = measurement(0);
  const double phi = measurement(1);

  Eigen::VectorXd mean(Model::kStateSize);
  mean << rho * cos(phi), rho * sin(phi), 0, 0, 0;

  Eigen::MatrixXd cov =
      Eigen::MatrixXd::Identity(Model::kStateSize, Model::kStateSize);
  cov(0, 0) = measurement_cov(0, 0);
  cov(1, 1) = measurement_cov(1, 1);

  return {mean, cov};
}

Eigen::VectorXd Radar::ApplyModel(const Eigen::VectorXd& pred_sigma_pt) {
  const double px = pred_sigma_pt(0);
  const double py = pred_sigma_pt(1);
  const double v = pred_sigma_pt(2);
  const double psi = pred_sigma_pt(3);
  // not used const double psi_dot = pred_sigma_pt(4);

  const double rho = sqrt(px * px + py * py);

  // TODO(dukexar): rho != 0
  // TODO(dukexar): px != 0
  const double phi = atan2(py, px);
  const double rho_dot = (px * cos(psi) * v + py * sin(psi) * v) / rho;

  Eigen::VectorXd result(Size());
  result << rho, phi, rho_dot;

  return result;
}

void Radar::NormalizeDelta(Eigen::VectorXd& delta) {
  delta(1) = Details::NormalizeAngle(delta(1));
}

Gaussian Laser::CreateInitialGaussian(const Eigen::VectorXd& measurement,
                                      const Eigen::MatrixXd& measurement_cov) {
  Eigen::VectorXd mean(Model::kStateSize);
  mean << measurement(0), measurement(1), 0, 0, 0;

  Eigen::MatrixXd cov =
      Eigen::MatrixXd::Identity(Model::kStateSize, Model::kStateSize);
  cov(0, 0) = measurement_cov(0, 0);
  cov(1, 1) = measurement_cov(1, 1);

  return {mean, cov};
}

Eigen::VectorXd Laser::ApplyModel(const Eigen::VectorXd& pred_sigma_pt) {
  const double px = pred_sigma_pt(0);
  const double py = pred_sigma_pt(1);

  Eigen::VectorXd result(Size());
  result << px, py;
  return result;
}

void Laser::NormalizeDelta(Eigen::VectorXd& delta) {}

}  // namespace Model