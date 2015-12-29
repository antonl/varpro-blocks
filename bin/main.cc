#include <iostream>
#include <armadillo>
#include "varpro-block.h"
#include "spdlog/spdlog.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
    auto logger = spdlog::get("varpro-blocks");
    logger->set_level(spdlog::level::debug);
    vec y = randu<vec>(15);
    vec t = linspace<vec>(0, 50, 15);
    vec p0 = {0.05};
    single_exp_model Z2(y, t);
    Z2.update_model(p0, true);

    cout << "Amat: \n" << Z2.Amat << endl;
    cout << "mjac: \n" << Z2.mjac << endl;
      
    return 0;
}
