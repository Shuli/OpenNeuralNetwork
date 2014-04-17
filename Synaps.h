/*!
 * @brief  Define synapses of neurons. This holds the ignition and weight.
 * @author Hisashi<ikari@akane.waseda.jp>, Waseda University. 
 * @file
 */
#ifndef NEURAL_SYNAPS_H_INCLUDED
#define NEURAL_SYNAPS_H_INCLUDED

#include <vector>

namespace neuralnet {

/*!
 * @brief Definition of a "SINGLE" synapse.
 */
template <typename V, class R>
class Synaps
{
public:
    Synaps(const R& parent, const unsigned int num);

    typedef V Value;
    const Value forwardOutput() const { return forwardOutput_; }
    void forwardOutput(Value value) { forwardOutput_ = value; }

    typedef std::vector<V> Values;  
    const Values& forwardWeights() { return forwardWeights_; }

    void forwardWeights(Values& values) { forwardWeights_ = values; }
    void forwardWeights(const unsigned int index, Value value) { forwardWeights_[index] = value; }
      
    const Value forwardFiring() { return forwardFiring_; }
    void forwardFiring(Value value) { forwardFiring_ = value; }

    const Value historyOfReverseFiring() { return historyOfReverseFiring_; }
    void historyOfReverseFiring(Value value) { historyOfReverseFiring_ = value; }

    Values& historyOfReverseWeights() { return historyOfReverseWeights_; }
    void historyOfReverseWeights(Values& values) { historyOfReverseWeights_ = values; }

    const Value reverseLearn() { return reverseLearn_; }
    void reverseLearn(Value value) { reverseLearn_ = value; }

protected:
    const R& parent_;

    Value  forwardOutput_;
    Values forwardWeights_;
    Value  forwardFiring_;
    Value  reverseLearn_;

    Value  historyOfReverseFiring_;
    Values historyOfReverseWeights_;
 
};

}; //end of namespace

#endif


