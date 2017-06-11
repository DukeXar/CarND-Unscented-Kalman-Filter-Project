#pragma once

#include <functional>
#include <iostream>
#include <type_traits>
#include "Eigen/Dense"

namespace Model {

const size_t kStateSize = 5;

struct Gaussian {
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

class CTRVUnscentedKalmanFilter {
 public:
  CTRVUnscentedKalmanFilter(double std_a, double std_yawdd,
                            Gaussian initialState);

  void Predict(double dt);

  template <typename Sensor>
  void Update(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& cov);

  Gaussian state() const { return state_; }
  double nis() const { return nis_; }

 private:
  const double std_a_;
  const double std_yawdd_;
  Gaussian state_;
  double nis_;
  Eigen::MatrixXd sigma_pts_;
};

class Radar {
 public:
  static constexpr size_t Size() { return 3; }

  // Returns radar sigma point from predicted state sigma point
  static Eigen::VectorXd ApplyModel(const Eigen::VectorXd& pred_sigma_pt);

  static Gaussian CreateInitialGaussian(const Eigen::VectorXd& measurement,
                                        const Eigen::MatrixXd& measurement_cov);

  static void NormalizeDelta(Eigen::VectorXd& delta);
};

class Laser {
 public:
  static constexpr size_t Size() { return 2; }

  // Returns laser sigma point from predicted state sigma point
  static Eigen::VectorXd ApplyModel(const Eigen::VectorXd& pred_sigma_pt);

  static Gaussian CreateInitialGaussian(const Eigen::VectorXd& measurement,
                                        const Eigen::MatrixXd& measurement_cov);

  static void NormalizeDelta(Eigen::VectorXd& delta);
};

namespace Details {

double NormalizeAngle(double angle);

Eigen::MatrixXd GenerateSigmaWeights(size_t sz);

Eigen::MatrixXd GenerateSigmaPoints(const Gaussian& state);

Eigen::MatrixXd CTRVGenerateAugmentedSigmaPoints(const Gaussian& state,
                                                 double std_a,
                                                 double std_yawdd);

Eigen::MatrixXd CTRVPredictSigmaPoints(const Eigen::MatrixXd& aug_sigma_pts,
                                       double dt);

void CTRVNormalizeStateAngle(Eigen::VectorXd& state);

Gaussian SigmaPointsToGaussian(
    const Eigen::MatrixXd& pred_sigma_pts,
    const std::function<void(Eigen::VectorXd&)>& normalizer);

struct KalmanUpdateResult {
  Gaussian state;
  double nis;
};

KalmanUpdateResult KalmanUpdate(
    const Gaussian& pred_state, const Eigen::MatrixXd& pred_state_sigma_pts,
    const Gaussian& pred_measurement,
    const Eigen::MatrixXd& pred_measurement_sigma_pts,
    const Eigen::VectorXd& measurement_mean);

template <typename Sensor>
inline Eigen::MatrixXd ApplySensorModel(const Eigen::MatrixXd& pred_sigma_pts) {
  Eigen::MatrixXd result(Sensor::Size(), pred_sigma_pts.cols());
  for (size_t i = 0; i < pred_sigma_pts.cols(); ++i) {
    result.col(i) = Sensor::ApplyModel(pred_sigma_pts.col(i));
  }
  return result;
}
}  // namespace Details

template <typename Sensor>
inline void CTRVUnscentedKalmanFilter::Update(
    const Eigen::VectorXd& measurement, const Eigen::MatrixXd& cov) {
  // TODO(dukexar): Would it be better to just leave pred_measurement creation
  // completely to measurement model and have a single call to it?
  using namespace Details;
  auto pred_measurement_sigma_pts = ApplySensorModel<Sensor>(sigma_pts_);
  auto pred_measurement = SigmaPointsToGaussian(pred_measurement_sigma_pts,
                                                &Sensor::NormalizeDelta);
  pred_measurement.cov += cov;
  auto res = KalmanUpdate(state_, sigma_pts_, pred_measurement,
                          pred_measurement_sigma_pts, measurement);
  state_ = res.state;
  nis_ = res.nis;
  std::cout << "After Update" << std::endl;
  std::cout << state_.mean << std::endl;
}

}  // namespace Model