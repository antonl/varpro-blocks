#include "test_templating.h"


int main() {
    std::vector<int> inp = {5};
    ctype_base A(1);
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
