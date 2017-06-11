#include "ukf.h"

#include <iostream>
#include <stdexcept>

namespace {
// The values look good from the NIS validation perspective.
// Tried to lower them, but after going lower than 2, the lower left quadrant
// in the dataset 1 is not processed correctly (noisy radar measurement make
// whole filter go wild for a while).
// Process noise standard deviation longitudinal acceleration in m/s^2
const double kStdA = 2;
// Process noise standard deviation yaw acceleration in rad/s^2
const double kStdYawdd = 2;

// To scale timestamp to seconds
const double kMicrosecondsInSecond = 1000000.0;
}  // namespace

UKF::UKF() : laser_cov_(2, 2), radar_cov_(3, 3), prev_timestamp_(0) {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

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
        Model::Gaussian initialState = Model::Laser::CreateInitialGaussian(
            meas_package.raw_measurements_, laser_cov_);
        ukf_.reset(new Model::CTRVUnscentedKalmanFilter(kStdA, kStdYawdd,
                                                        initialState));
        break;
      }
      case MeasurementPackage::SensorType::RADAR: {
        Model::Gaussian initialState = Model::Radar::CreateInitialGaussian(
            meas_package.raw_measurements_, radar_cov_);
        ukf_.reset(new Model::CTRVUnscentedKalmanFilter(kStdA, kStdYawdd,
                                                        initialState));
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

  ukf_->Predict(dt);

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