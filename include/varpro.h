#pragma once

#include "CXX/Extensions.hxx"
#include <memory>

template<typename T> class ResponseBlock<T> : 
    public Py::PythonExtension<ResponseBlock<T>>
{
public:
    ResponseBlock();
    virtual ~ResponseBlock();
    static void init_type(void);

    virtual Py::Object repr();
private:
    typedef T value;
    std::unique_ptr<T> ptr;
};
