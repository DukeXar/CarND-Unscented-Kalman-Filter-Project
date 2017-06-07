#include "model.h"

#include <cmath>
#include <stdexcept>

namespace Model {

const size_t kStateSize = 5;
const size_t kNoiseCovSize = 2;
const size_t kAugStateSize = kStateSize + kNoiseCovSize;

Eigen::MatrixXd GenerateSigmaPoints(const Gaussian& state) {
  const size_t n_x = state.mean.rows();
  const double lambda = 3 - n_x;

  const Eigen::MatrixXd sqrt_cov = state.cov.llt().matrixL();

  Eigen::MatrixXd result(n_x, 2 * n_x + 1);
  result.col(0) = state.mean;

  for (size_t i = 1; i < n_x; ++i) {
    const auto k = sqrt(lambda + n_x) * sqrt_cov.col(i);
    result.col(i) = state.mean + k;
    result.col(i + n_x) = state.mean - k;
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
  if (fabs(psi) > 0.001) {
    px_new = px / psi_dot * (sin(psi + psi_dot * dt) - sin(psi));
    py_new = py / psi_dot * (cos(psi) - cos(psi + psi_dot * dt));
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
  while (angle < -M_PI) {
    angle += 2 * M_PI;
  }
  while (angle > M_PI) {
    angle -= 2 * M_PI;
  }
  return angle;
}

Eigen::MatrixXd GenerateSigmaWeights(size_t sz) {
  const double lambda = 3 - sz;

  Eigen::VectorXd weights(sz);
  weights(0) = lambda / (lambda + sz);
  for (size_t i = 1; i < weights.rows(); ++i) {
    weights(i) = 0.5 / (lambda + sz);
  }

  return weights;
}

Gaussian SigmaPointsToGaussian(const Eigen::MatrixXd& pred_sigma_pts) {
  const size_t n_sigma = (pred_sigma_pts.cols() - 1) / 2;
  if (n_sigma != kAugStateSize) {
    throw std::runtime_error("Invalid input for SigmaPointsToState");
  }

  Eigen::VectorXd weights = GenerateSigmaWeights(pred_sigma_pts.cols());

  Eigen::VectorXd res_x(kStateSize);
  res_x.fill(0);
  for (size_t i = 1; i < n_sigma; ++i) {
    res_x = res_x + weights(i) * pred_sigma_pts.col(i);
  }

  Eigen::MatrixXd res_cov(kStateSize, kStateSize);
  res_cov.fill(0);
  for (size_t i = 0; i < weights.rows(); ++i) {
    Eigen::VectorXd diff = pred_sigma_pts.col(i) - res_x;
    diff(3) = NormalizeAngle(diff(3));
    res_cov = res_cov + weights(i) * diff * diff.transpose();
  }

  return {res_x, res_cov};
}

Eigen::MatrixXd CreateCrossCorrelation(
    const Eigen::VectorXd& state_mean, const Eigen::MatrixXd& state_sigma_pts,
    const Eigen::VectorXd& measurement_mean,
    const Eigen::MatrixXd& measurement_sigma_pts) {
  Eigen::VectorXd weights = GenerateSigmaWeights(state_sigma_pts.cols());
  Eigen::MatrixXd result(state_mean.rows(), measurement_mean.rows());
  result.fill(0);
  for (size_t i = 0; i < weights.rows(); ++i) {
    auto state_diff = (state_sigma_pts.col(i) - state_mean);
    auto measurement_diff = (measurement_sigma_pts.col(i) - measurement_mean);
    result.col(i) += weights(i) * state_diff * measurement_diff.transpose();
  }
  return result;
}

Gaussian KalmanUpdate(const Gaussian& pred_state,
                      const Eigen::MatrixXd& pred_state_sigma_pts,
                      const Gaussian& pred_measurement,
                      const Eigen::MatrixXd& pred_measurement_sigma_pts,
                      const Eigen::VectorXd& measurement_mean) {
  Eigen::MatrixXd cross_corr =
      CreateCrossCorrelation(pred_state.mean, pred_state_sigma_pts,
                             pred_measurement.mean, pred_measurement_sigma_pts);

  Eigen::MatrixXd kalman_gain = cross_corr * pred_measurement.cov.inverse();

  Eigen::VectorXd new_state_mean =
      pred_state.mean +
      kalman_gain * (measurement_mean - pred_measurement.mean);

  Eigen::MatrixXd new_state_cov =
      pred_state.cov -
      kalman_gain * pred_measurement.cov * kalman_gain.transpose();
  return {new_state_mean, new_state_cov};
}

UnscentedKalmanFilter::UnscentedKalmanFilter(double std_a, double std_yawdd)
    : std_a_(std_a),
      std_yawdd_(std_yawdd),
      state_{Eigen::VectorXd::Zero(kStateSize),
             Eigen::MatrixXd::Identity(kStateSize, kStateSize)} {}

void UnscentedKalmanFilter::Predict(double dt) {
  sigma_pts_ = CTRVPredictSigmaPoints(
      CTRVGenerateAugmentedSigmaPoints(state_, std_a_, std_yawdd_), dt);
  state_ = SigmaPointsToGaussian(sigma_pts_);
}

Eigen::VectorXd Radar::ApplyModel(const Eigen::VectorXd& pred_sigma_pt) {
  const double px = pred_sigma_pt(0);
  const double py = pred_sigma_pt(1);
  const double v = pred_sigma_pt(2);
  const double psi = pred_sigma_pt(3);
  const double psi_dot = pred_sigma_pt(4);

  const double rho = sqrt(px * px + py * py);

  // TODO(dukexar): rho != 0
  // TODO(dukexar): px != 0
  const double phi = atan2(py, px);
  const double rho_dot = (px * cos(phi) * v + py * sin(phi) * v) / rho;

  Eigen::VectorXd result(Size());
  result << rho, phi, rho_dot;
  return result;
}

Gaussian Radar::PredictMeasurementGaussian(
    const Eigen::MatrixXd& pred_sensor_sigma_pts, const Eigen::MatrixXd& cov) {
  const size_t n_sigma = (pred_sensor_sigma_pts.cols() - 1) / 2;

  // TODO(dukexar): Can weights be shared?
  Eigen::VectorXd weights = GenerateSigmaWeights(pred_sensor_sigma_pts.cols());

  Eigen::VectorXd res_x(weights.rows());
  res_x.fill(0);
  for (size_t i = 1; i < n_sigma; ++i) {
    res_x = res_x + weights(i) * pred_sensor_sigma_pts.col(i);
  }

  Eigen::MatrixXd res_cov(weights.rows(), weights.rows());
  res_cov.fill(0);
  for (size_t i = 0; i < weights.rows(); ++i) {
    Eigen::VectorXd diff = pred_sensor_sigma_pts.col(i) - res_x;
    diff(1) = NormalizeAngle(diff(1));
    res_cov.col(i) = weights(i) * diff * diff.transpose();
  }
  res_cov += cov;

  return {res_x, res_cov};
}

Eigen::VectorXd Laser::ApplyModel(const Eigen::VectorXd& pred_sigma_pt) {
  const double px = pred_sigma_pt(0);
  const double py = pred_sigma_pt(1);

  Eigen::VectorXd result(Size());
  result << px, py;
  return result;
}

Gaussian Laser::PredictMeasurementGaussian(
    const Eigen::MatrixXd& pred_sensor_sigma_pts, const Eigen::MatrixXd& cov) {
  const size_t n_sigma = (pred_sensor_sigma_pts.cols() - 1) / 2;
  Eigen::VectorXd weights = GenerateSigmaWeights(pred_sensor_sigma_pts.cols());

  Eigen::VectorXd res_x(weights.rows());
  res_x.fill(0);
  for (size_t i = 1; i < n_sigma; ++i) {
    res_x = res_x + weights(i) * pred_sensor_sigma_pts.col(i);
  }

  Eigen::MatrixXd res_cov(weights.rows(), weights.rows());
  res_cov.fill(0);
  for (size_t i = 0; i < weights.rows(); ++i) {
    Eigen::VectorXd diff = pred_sensor_sigma_pts.col(i) - res_x;
    res_cov.col(i) = weights(i) * diff * diff.transpose();
  }
  res_cov += cov;

  return {res_x, res_cov};
}

}  // namespace Model