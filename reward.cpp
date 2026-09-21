#include <cmath>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

double pose_similarity_reward(const DoubleArray& current,
                              const DoubleArray& reference) {
    const py::buffer_info current_info = current.request();
    const py::buffer_info reference_info = reference.request();
    if (current_info.ndim != 1 || reference_info.ndim != 1) {
        throw std::invalid_argument("current and reference must be 1-D arrays");
    }
    if (current_info.shape[0] != reference_info.shape[0]) {
        throw std::invalid_argument("current and reference must have the same length");
    }

    const auto* current_data = static_cast<const double*>(current_info.ptr);
    const auto* reference_data = static_cast<const double*>(reference_info.ptr);
    double squared_error = 0.0;
    for (py::ssize_t i = 0; i < current_info.shape[0]; ++i) {
        const double difference = current_data[i] - reference_data[i];
        squared_error += difference * difference;
    }
    return std::exp(-2.0 * squared_error);
}

PYBIND11_MODULE(reward_cpp, module) {
    module.def(
        "pose_similarity_reward",
        &pose_similarity_reward,
        py::arg("current"),
        py::arg("reference"));
}
