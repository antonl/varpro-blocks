#pragma once

#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include "spdlog/spdlog.h"
#include <memory>

class arma_module : public Py::ExtensionModule<arma_module>
{
public:
    arma_module();
    virtual ~arma_module();

    Py::Object vec_factory(const Py::Tuple &args);
private:
    std::shared_ptr<spdlog::logger> logger;
};
