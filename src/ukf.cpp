#include "ukf.h"

#include <iostream>
#include <stdexcept>

namespace {
// Process noise standard deviation longitudinal acceleration in m/s^2
const double kStdA = 30;
// Process noise standard deviation yaw acceleration in rad/s^2
const double kStdYawdd = 30;
// To scale timestamp to second
const double kMicrosecondsInSecond = 1000000.0;
}  // namespace

UKF::UKF() : laser_cov_(2, 2), radar_cov_(3, 3), prev_timestamp_(0) {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  laser_cov_.fill(0);
  // Laser measurement noise standard deviation position1 in m
  laser_cov_(0, 0) = 0.15 * 0.15;
  // Laser measurement noise standard deviation position2 in m
  laser_cov_(1, 1) = 0.15 * 0.15;

  radar_cov_.fill(0);
  // Radar measurement noise standard deviation radius in m
  radar_cov_(0, 0) = 0.3 * 0.3;
  // Radar measurement noise standard deviation angle in rad
  radar_cov_(1, 1) = 0.03 * 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  radar_cov_(2, 2) = 0.3 * 0.3;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // TODO(dukexar): Can we live without switch here? Too ugly to have a template
  // and a switch together...
  if (!ukf_) {
    switch (meas_package.sensor_type_) {
      case MeasurementPackage::SensorType::LASER: {
        if (use_laser_) {
          Model::Gaussian initialState{
              Model::Laser::CreateFirstMean(meas_package.raw_measurements_),
              Eigen::MatrixXd::Identity(Model::kStateSize, Model::kStateSize)};
          ukf_.reset(new Model::CTRVUnscentedKalmanFilter(kStdA, kStdYawdd,
                                                          initialState));
        }
        break;
      }
      case MeasurementPackage::SensorType::RADAR: {
        if (use_radar_) {
          Model::Gaussian initialState{
              Model::Radar::CreateFirstMean(meas_package.raw_measurements_),
              Eigen::MatrixXd::Identity(Model::kStateSize, Model::kStateSize)};
          ukf_.reset(new Model::CTRVUnscentedKalmanFilter(kStdA, kStdYawdd,
                                                          initialState));
        }
        break;
      }
      default:
        throw std::runtime_error("Invalid sensor type");
    }

    prev_timestamp_ = meas_package.timestamp_;
    return;
  }

  double dt =
      (meas_package.timestamp_ - prev_timestamp_) / kMicrosecondsInSecond;
  std::cout << ">>> Predicting dt=" << dt << std::endl;
  std::cout << "state=\n"
            << ukf_->state().mean << "\n---\n"
            << ukf_->state().cov << "\n---" << std::endl;
  ukf_->Predict(dt);

  std::cout << ">>> Predicted" << std::endl;
  std::cout << "state=\n"
            << ukf_->state().mean << "\n---\n"
            << ukf_->state().cov << "\n---" << std::endl;

  std::cout << ">>> Updating " << meas_package.sensor_type_ << std::endl;
  switch (meas_package.sensor_type_) {
    case MeasurementPackage::SensorType::LASER: {
      if (use_laser_) {
        ukf_->Update<Model::Laser>(meas_package.raw_measurements_, laser_cov_);
      }
      break;
    }
    case MeasurementPackage::SensorType::RADAR: {
      if (use_radar_) {
        ukf_->Update<Model::Radar>(meas_package.raw_measurements_, radar_cov_);
      }
      break;
    }
    default:
      throw std::runtime_error("Invalid sensor type");
  }

  prev_timestamp_ = meas_package.timestamp_;
}