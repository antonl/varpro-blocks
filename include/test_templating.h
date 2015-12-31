#pragma once

#include <vector>
#include <iostream>
#include <type_traits>

using std::cout;
using std::endl;

class ctype_base
{
public:
    explicit ctype_base(int x) : m_x(x) {
        cout << "constructed ctype_base(" << x << ")" << endl;
    }
    virtual ~ctype_base() {
        cout << "destroyed ctype_base(" << m_x << ")" << endl;
    }

private:
    ctype_base() 
    {
        cout << "ctype_base default construction" << endl;
    }
protected:
    int m_x;

};

class ctype_child: public ctype_base
{
public:
    explicit ctype_child(int x, int y) : ctype_base(x), m_y(y) {
        cout << "constructed ctype_child(" << x << ", " << y << ")" << endl;
    }
    virtual ~ctype_child() {
        cout << "destroyed ctype_child(" << m_x << ", " << m_y << ")" << endl;
    }
private:
    ctype_child():ctype_base(0)
    {
        cout << "ctype_child default construction" << endl;
    }

protected:
    int m_y;
};

template <typename T> struct wrapper
{
    // do not wrap other types with this template
    static_assert(std::is_base_of<ctype_base, T>::value, "wrapper only works on derived types of ctype_base");

    explicit wrapper(const std::vector<int> inp) :
        m_ctype(inp[0])
    {
    }

    T m_ctype;
};

template <> wrapper<ctype_child>::wrapper(std::vector<int> inp) :
    m_ctype(inp[0], inp[1])
{
}
