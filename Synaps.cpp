/*!
 * @brief  The actual processing of the synapse.
 * @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.
 * @file
 */
#include <complex>
#include <stdlib.h>
#include "Synaps.h"
#include "Neuron.h"
#include "Prim.h"

namespace neuralnet {

/*!
 * @brief The actual processing of the synapse.
 */
template <typename U>
class SynapsImpl
{
public:
    static const U rand003();
};


template <typename V, class R>
Synaps<V, R>::Synaps(const R& parent, const unsigned int num) : parent_(parent)
{
    forwardWeights_.resize(num);
    historyOfReverseWeights_.resize(num);

    for (unsigned int i = 0; i < num; i++) {
        forwardWeights_[i] = SynapsImpl<V>::rand003();
        historyOfReverseWeights_[i] = Prim<V>::min();
    }
}


template <typename U>
const U SynapsImpl<U>::rand003()
{
    return (rand() % 10) * 0.06 - 0.3;
}


template <>
const std::complex<double> SynapsImpl< std::complex<double> >::rand003()
{
    return std::complex<double>((rand() % 10) * 0.06 - 0.3,
                                (rand() % 10) * 0.06 - 0.3);
}


/*!
 * @brief Create an instance of the double type externally provided.
 */
template class Synaps< double, Neuron<double> >;


/*!
 * @brief Create an instance of the double type externally provided.
 */
template class Synaps< std::complex<double>, Neuron< std::complex<double> > >;


}; // end of namespace



