/*!
 * @brief  Provides maximum value and the initial value of the number.
 * @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.
 * @file
 */
#include <complex>
#include "Prim.h"

namespace neuralnet {

template <typename V>
const V Prim<V>::min()
{
    return 0.0;
}


template <>
const std::complex<double> Prim< std::complex<double> >::min()
{
    return std::complex<double>(0.0, 0.0);
}


template <typename V>
const V Prim<V>::max()
{
    return 1.0;
}


template <>
const std::complex<double> Prim< std::complex<double> >::max()
{
    return std::complex<double>(1.0, 1.0);
}


/*!
 * @brief Create an instance of the double type externally provided.
 */
template class Prim<double>;


/*!
 * @brief Create an instance of the complex type externally provided.
 */
template class Prim< std::complex<double> >;



}; // end of namespace



