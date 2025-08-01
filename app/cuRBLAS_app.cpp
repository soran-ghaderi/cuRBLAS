#include "cuRBLAS/cuRBLAS.hpp"

#include <iostream>

int main(){
  int result = cuRBLAS::add_one(1);
//  cuRBLAS::reduceSum()
  std::cout << "1 + 1 = " << result << std::endl;
}