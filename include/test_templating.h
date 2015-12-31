#pragma once

#include <vector>
#include <iostream>
#include <type_traits>

using std::cout;
using std::endl;

class ctype_base
{
public:
    explicit ctype_base(int x);
    virtual ~ctype_base();    
    constexpr const static auto name = "ctype_base";

private:
    ctype_base(); 
protected:
    int m_x;
};

class ctype_child: public ctype_base
{
public:
    explicit ctype_child(int x, int y);
    virtual ~ctype_child();
    constexpr const static auto name = "ctype_child";
private:
    ctype_child();

protected:
    int m_y;
};

template <typename T> struct wrapper
{
    explicit wrapper(const std::vector<int> inp);
    T m_ctype;
};
