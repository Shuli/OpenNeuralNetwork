/*!
 * @brief  Provides maximum value and the initial value of the number.
 * @author Hisashi<ikari@akane.waseda.jp>, Waseda University. 
 * @file
 */
#ifndef NEURAL_VALUE_H_INCLUDED
#define NEURAL_VALUE_H_INCLUDED

namespace neuralnet {

/*!
 * @brief Provides maximum value and the initial value of the number.
 */
template <typename V>
class Prim
{
public:
    static const V min();
    static const V max();
};

}; //end of namespace

#endif


