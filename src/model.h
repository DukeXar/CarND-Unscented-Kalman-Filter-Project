#pragma once

#include "Eigen/Dense"

#include <cmath>

namespace Model {
Eigen::MatrixXd GenerateSigmaPoints(const Eigen::VectorXd& state,
                                    const Eigen::MatrixXd& cov);

Eigen::MatrixXd GenerateAugmentedSigmaPoints(const Eigen::VectorXd& state,
                                             const Eigen::MatrixXd& cov,
                                             double std_a, double std_yawdd);

Eigen::MatrixXd CTRVPredict(const Eigen::MatrixXd& aug_sigma_pts, double dt);
}