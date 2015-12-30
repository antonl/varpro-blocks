#include "CXX/Objects.hxx"
#include "CXX/Extensions.hxx"
#include <iostream>

template<typename T> ResponseBlock<T>::ResponseBlock() {
    std::cout << "generic response block ctor" << std::endl;
}

template<typename T> ResponseBlock<T>::~ResponseBlock() {
    std::cout << "generic response block dtor" << std::endl;
}

template<typename T> static void ResponseBlock<T>::init_type(void) {
    std::cout << "initializing response block" << std::endl;
}

class varpro_module : public Py::ExtensionModule<varpro_module>
{
public:
    varpro_module() : Py::ExtensionModule<varpro_module>("varpro") {
        initialize("module for variable projection");
    }

    virtual ~varpro_module() {
    }
};

#if defined( _WIN32 )
#define EXPORT_SYMBOL __declspec( dllexport )
#else
#define EXPORT_SYMBOL
#endif

extern "C" EXPORT_SYMBOL PyObject *PyInit_varpro() {
    static varpro_module *instance = new varpro_module;
    std::cout << "Initializing module varpro" << std::endl;
    return instance->module().ptr();
}

// symbol required for the debug version
extern "C" EXPORT_SYMBOL PyObject *PyInit_varpro_d() { 
    return PyInit_varpro();
}
