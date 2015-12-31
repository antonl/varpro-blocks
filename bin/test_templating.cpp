#include "test_templating.h"
#include <vector>
#include <iostream>
#include <type_traits>

using std::cout;
using std::endl;

ctype_base::ctype_base(int x): 
    m_x(x) 
{
    cout << "constructed ctype_base(" << x << ")" << endl;
}
ctype_base::~ctype_base() {
    cout << "destroyed ctype_base(" << m_x << ")" << endl;
}

ctype_base::ctype_base() 
{
    cout << "ctype_base default construction" << endl;
}

ctype_child::ctype_child(int x, int y):
    ctype_base(x),
    m_y(y) 
{
    cout << "constructed ctype_child(" << x << ", " << y << ")" << endl;
}
ctype_child::~ctype_child() 
{
    cout << "destroyed ctype_child(" << m_x << ", " << m_y << ")" << endl;
}

ctype_child::ctype_child():
    ctype_base(0)
{
    cout << "ctype_child default construction" << endl;
}

template <typename T> wrapper<T>::wrapper(const std::vector<int> inp) :
    m_ctype(inp[0])
{
    static_assert(std::is_base_of<ctype_base, T>::value, "wrapper only works on derived types of ctype_base");
    cout << "constructed wrapper<" << T::name << ">" << endl;
}

template <> wrapper<ctype_child>::wrapper(std::vector<int> inp) :
    m_ctype(inp[0], inp[1])
{
    cout << "constructed wrapper<" << ctype_child::name << ">" << endl;
}


int main() {
    std::vector<int> inp = {5};
    ctype_base A(1);
    cout << "A name: " << A.name << endl;
    ctype_child B(2, 3);
    wrapper<ctype_base> a(inp);
    std::vector<int> inp2 = {6, 7};
    wrapper<ctype_child> b(inp2);


    /*
    cout << "constructing A2" << endl;
    ctype_base A2();
    cout << "done" << endl;
    */

    //wrapper<double> c(0.1); // error!
}
