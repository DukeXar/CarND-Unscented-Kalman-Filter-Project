#pragma once

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

  Eigen::VectorXd state() const { return ukf_.state().mean; }

 private:
  Model::UnscentedKalmanFilter ukf_;

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Weights of sigma points
  Eigen::VectorXd weights_;

  Eigen::MatrixXd laser_cov_;
  Eigen::MatrixXd radar_cov_;
};