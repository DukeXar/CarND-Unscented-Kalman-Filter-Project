#include "model.h"

#include <cmath>
#include <stdexcept>

namespace Model {
Eigen::MatrixXd GenerateSigmaPoints(const Eigen::VectorXd& state,
                                    const Eigen::MatrixXd& cov) {
  const size_t n_x = state.rows();
  const double lambda = 3 - n_x;

  const Eigen::MatrixXd sqrt_cov = cov.llt().matrixL();

  Eigen::MatrixXd result(n_x, 2 * n_x + 1);
  result.col(0) = state;

  for (size_t i = 1; i < n_x; ++i) {
    const auto k = sqrt(lambda + n_x) * sqrt_cov.col(i);
    result.col(i) = state + k;
    result.col(i + n_x) = state - k;
  }

  return result;
}

Eigen::MatrixXd GenerateAugmentedSigmaPoints(const Eigen::VectorXd& state,
                                             const Eigen::MatrixXd& cov,
                                             double std_a, double std_yawdd) {
  const size_t n_x = state.rows();
  const size_t aug_n_x = n_x + 2;

  Eigen::VectorXd aug_state(aug_n_x);
  aug_state.fill(0);
  aug_state.head(n_x) = state;

  Eigen::MatrixXd aug_cov(aug_n_x, aug_n_x);
  aug_cov.fill(0);
  aug_cov.topLeftCorner(n_x, n_x) = cov;
  aug_cov(n_x, n_x) = std_a * std_a;
  aug_cov(n_x + 1, n_x + 1) = std_yawdd * std_yawdd;

  return GenerateSigmaPoints(aug_state, aug_cov);
}

Eigen::VectorXd CTRVPredictVector(const Eigen::VectorXd& aug_sigma_pts, double dt) {
  if (aug_sigma_pts.rows() != 7) {
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

  const double px_new_noise = 0.5 * dt*dt * nu_a * cos(psi);
  const double py_new_noise = 0.5 * dt*dt * nu_a * sin(psi);

  const double v_new_noise = dt * nu_a;

  const double psi_new_noise = 0.5 * dt*dt * nu_psi_dotdot;
  const double psi_dot_new_noise = dt * nu_psi_dotdot;

  Eigen::VectorXd result(5);
  result(0) = px_new + px_new_noise;
  result(1) = py_new + py_new_noise;
  result(2) = v + v_new_noise;
  result(3) = psi_new + psi_new_noise;
  result(4) = psi_dot + psi_dot_new_noise;

  return result;
}

Eigen::MatrixXd CTRVPredict(const Eigen::MatrixXd& aug_sigma_pts, double dt) {
  if (aug_sigma_pts.rows() != 7) {
    throw std::runtime_error("Invalid input");
  }

  Eigen::MatrixXd result(5, aug_sigma_pts.cols());

  for (size_t i = 0; i < aug_sigma_pts.cols(); ++i) {
    result.col(i) = CTRVPredictVector(aug_sigma_pts.col(i), dt);
  }

  return result;
}
}  // namespace Model