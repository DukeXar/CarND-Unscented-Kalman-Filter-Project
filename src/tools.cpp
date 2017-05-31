#include "tools.h"

#include <stdexcept>

Eigen::VectorXd Tools::CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                     const std::vector<Eigen::VectorXd> &ground_truth) {

  if (estimations.size() != ground_truth.size()) {
    throw std::runtime_error("Estimations and ground truth must be of the same size");
  }

  if (estimations.empty()) {
    return Eigen::VectorXd(estimations[0].size());
  }

  Eigen::VectorXd sum = Eigen::VectorXd::Zero(estimations[0].size());
  for (size_t i = 0; i < estimations.size(); ++i) {
    Eigen::VectorXd sqr = (estimations[i] - ground_truth[i]).array().square();
    sum += sqr;
  }

  Eigen::VectorXd mean = sum / estimations.size();
  return mean.array().sqrt();
}
