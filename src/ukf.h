#pragma once

#include <memory>
#include "Eigen/Dense"
#include "measurement_package.h"
#include "model.h"

class UKF {
 public:
  UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  bool has_state() const { return (bool)ukf_; }
  Eigen::VectorXd state() const { return ukf_->state().mean; }
  double nis() const { return ukf_->nis(); }

 private:
  std::unique_ptr<Model::CTRVUnscentedKalmanFilter> ukf_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* time when the state is true, in us
  long long prev_timestamp_;

  ///* Weights of sigma points
  Eigen::VectorXd weights_;

  Eigen::MatrixXd laser_cov_;
  Eigen::MatrixXd radar_cov_;
};