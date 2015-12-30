#include "arma_module.h"
#include "arma_objects.h"
#include "numpy/arrayobject.h"

#include <iostream>

arma_module::arma_module() : Py::ExtensionModule<arma_module>("arma") 
{
    arma_mat::init_type();
    initialize("armadillo to numpy bridge");

    Py::Dict d(moduleDictionary());
    Py::Object mat(arma_mat::type());
    d["mat"] = mat;
}

arma_module::~arma_module() 
{
    std::cout << "deinit arma_module" << std::endl;
}

#if defined( _WIN32 )
#define EXPORT_SYMBOL __declspec( dllexport )
#else
#define EXPORT_SYMBOL
#endif

extern "C" EXPORT_SYMBOL PyObject *PyInit_arma() {
    import_array();
    static arma_module *instance = new arma_module;
    std::cout << "Initializing module arma" << std::endl;
    return instance->module().ptr();
}

// symbol required for the debug version
extern "C" EXPORT_SYMBOL PyObject *PyInit_arma_d() { 
    return PyInit_arma();
}
