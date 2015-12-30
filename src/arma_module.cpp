#include "arma_module.h"
#include "arma_objects.h"
#include "numpy/arrayobject.h"

#include <iostream>

arma_module::arma_module() : Py::ExtensionModule<arma_module>("arma"), 
    logger(spdlog::get("arma"))
{
    logger->debug("in arma_module ctor");
    arma_mat::init_type();
    arma_vec::init_type();
    initialize("armadillo to numpy bridge");

    Py::Dict d(moduleDictionary());
    Py::Object mat(arma_mat::type());
    Py::Object vec(arma_vec::type());
    d["mat"] = mat;
    d["vec"] = vec;
    logger->debug("added mat object to module dict");
}

arma_module::~arma_module() 
{
    logger->debug("in arma_module dtor");
}

#if defined( _WIN32 )
#define EXPORT_SYMBOL __declspec( dllexport )
#else
#define EXPORT_SYMBOL
#endif

extern "C" EXPORT_SYMBOL PyObject *PyInit_arma() {
    import_array();
    auto logger = spdlog::stdout_logger_mt("arma");
    logger->set_level(spdlog::level::debug);

    logger->debug("initializing module arma");
    static arma_module *instance = new arma_module;
    return instance->module().ptr();
}

// symbol required for the debug version
extern "C" EXPORT_SYMBOL PyObject *PyInit_arma_d() { 
    return PyInit_arma();
}