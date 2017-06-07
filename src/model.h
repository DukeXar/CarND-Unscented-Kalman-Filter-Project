#pragma once

#include <type_traits>
#include "Eigen/Dense"

namespace Model {

struct Gaussian {
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

// Returns sigma-points matrix from state and covariance matrix
Eigen::MatrixXd GenerateSigmaPoints(const Gaussian& state);

// Returns augmented sigma points matrix from state, covariance matrix and
// noises
Eigen::MatrixXd CTRVGenerateAugmentedSigmaPoints(const Gaussian& state,
                                                 double std_a,
                                                 double std_yawdd);

// Returns predicted sigma-points using augmented sigma points and time delta
Eigen::MatrixXd CTRVPredictSigmaPoints(const Eigen::MatrixXd& aug_sigma_pts,
                                       double dt);

// Returns gaussian from predicted sigma points
Gaussian SigmaPointsToGaussian(const Eigen::MatrixXd& pred_sigma_pts);

Eigen::MatrixXd CreateCrossCorrelation(
    const Eigen::VectorXd& state_mean, const Eigen::MatrixXd& state_sigma_pts,
    const Eigen::VectorXd& measurement_mean,
    const Eigen::MatrixXd& measurement_sigma_pts);

Gaussian KalmanUpdate(const Gaussian& pred_state,
                      const Eigen::MatrixXd& pred_state_sigma_pts,
                      const Gaussian& pred_measurement,
                      const Eigen::MatrixXd& pred_measurement_sigma_pts,
                      const Eigen::VectorXd& measurement_mean);

class UnscentedKalmanFilter {
 public:
  UnscentedKalmanFilter(double std_a, double std_yawdd);

  void Predict(double dt);

  template <typename Sensor>
  void Update(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& cov);

  Gaussian state() const { return state_; }

 private:
  const double std_a_;
  const double std_yawdd_;
  Gaussian state_;
  Eigen::MatrixXd sigma_pts_;
};

class Radar {
 public:
  static constexpr size_t Size() { return 3; }

  // Returns radar sigma point from predicted state sigma point
  static Eigen::VectorXd ApplyModel(const Eigen::VectorXd& pred_sigma_pt);

  static Gaussian PredictMeasurementGaussian(
      const Eigen::MatrixXd& pred_sensor_sigma_pts, const Eigen::MatrixXd& cov);
};

class Laser {
 public:
  static constexpr size_t Size() { return 2; }

  // Returns laser sigma point from predicted state sigma point
  static Eigen::VectorXd ApplyModel(const Eigen::VectorXd& pred_sigma_pt);

  static Gaussian PredictMeasurementGaussian(
      const Eigen::MatrixXd& pred_sensor_sigma_pts, const Eigen::MatrixXd& cov);
};

template <typename Sensor>
inline Eigen::MatrixXd ApplyModelToSigmaPoints(
    const Eigen::MatrixXd& pred_sigma_pts) {
  Eigen::MatrixXd result(Sensor::Size(), pred_sigma_pts.cols());
  for (size_t i = 0; i < pred_sigma_pts.cols(); ++i) {
    result.col(i) = Sensor::ApplyModel(pred_sigma_pts.col(i));
  }
  return result;
}

template <typename Sensor>
inline void UnscentedKalmanFilter::Update(const Eigen::VectorXd& measurement,
                                          const Eigen::MatrixXd& cov) {
  auto pred_measurement_sigma_pts = ApplyModelToSigmaPoints<Sensor>(sigma_pts_);
  auto pred_measurement =
      Sensor::PredictMeasurementGaussian(pred_measurement_sigma_pts, cov);
  state_ = KalmanUpdate(state_, sigma_pts_, pred_measurement,
                        pred_measurement_sigma_pts, measurement);
}
}