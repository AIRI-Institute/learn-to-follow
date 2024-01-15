// cppimport
#include "config.h"

PYBIND11_MODULE(config, m) {
py::class_<Config>(m, "Config")
    .def(py::init<>())
    .def_readwrite("use_static_cost", &Config::use_static_cost)
    .def_readwrite("use_dynamic_cost", &Config::use_dynamic_cost)
    .def_readwrite("reset_dynamic_cost", &Config::reset_dynamic_cost)
    .def_readwrite("use_rl", &Config::use_rl)
    .def_readwrite("obs_radius", &Config::obs_radius)
    .def_readwrite("num_threads", &Config::num_threads)
    .def_readwrite("seed", &Config::seed)
    .def_readwrite("path_to_weights", &Config::path_to_weights)
;
}
<%
setup_pybind11(cfg)
%>
