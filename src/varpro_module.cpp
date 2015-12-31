#include "varpro_module.h"
#include "varpro_objects.h"

varpro_module::varpro_module() : Py::ExtensionModule<varpro_module>("varpro"),
    logger(spdlog::get("varpro"))
{
    logger->debug("in varpro_module ctor");
    varpro_block<single_exp_block>::init_type();
    initialize("varpro c++ implementation");

    Py::Dict d(moduleDictionary());
    Py::Object single_exp(varpro_block<single_exp_block>::type());
    d["single_exp_block"] = single_exp;
    logger->debug("added single_exp_block to module dict");
}

varpro_module::~varpro_module()
{
    logger->debug("in varpro_module dtor");
}

#if defined( _WIN32 )
#define EXPORT_SYMBOL __declspec( dllexport )
#else
#define EXPORT_SYMBOL
#endif

extern "C" EXPORT_SYMBOL PyObject *PyInit_varpro() {
    auto logger1 = spdlog::stdout_logger_mt("varpro");
    auto logger2 = spdlog::stdout_logger_mt("varpro.cvarpro_block");
    auto logger3 = spdlog::stdout_logger_mt("varpro.varpro_block");

    logger1->set_level(spdlog::level::debug);
    logger2->set_level(spdlog::level::debug);
    logger3->set_level(spdlog::level::debug);

    logger1->debug("initializing module varpro");
    static varpro_module *instance = new varpro_module;
    return instance->module().ptr();
}

// symbol required for the debug version
extern "C" EXPORT_SYMBOL PyObject *PyInit_varpro_d() { 
    return PyInit_varpro();
}
