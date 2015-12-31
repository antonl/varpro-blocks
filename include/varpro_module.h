#pragma once

#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include "spdlog/spdlog.h"
#include <memory>

class varpro_module : public Py::ExtensionModule<varpro_module>
{
public:
    varpro_module();
    virtual ~varpro_module();
private:
    std::shared_ptr<spdlog::logger> logger;
};
