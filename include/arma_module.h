#pragma once

#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"

class arma_module : public Py::ExtensionModule<arma_module>
{
public:
    arma_module();
    virtual ~arma_module();
};
