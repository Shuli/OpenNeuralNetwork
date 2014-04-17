/*!
 * @brief  It will answers by a back propagation.
 * @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.
 * @file
 */
#include <string>
#include <vector>
#include <complex>
#include <cassert>

#include "Synaps.h"
#include "Neuron.h"
#include "Prim.h"

namespace neuralnet {

/*!
 * @brief Judgment and learning of neural networks by the back propagation.
 */
template <typename U>
class NeuronImpl
{
public:
    typedef std::vector< Synaps<U, Neuron<U> >* > Synapses;
    static void learnWeight(const Neuron<U>* parent, const Synapses& synapses, unsigned int index);
    static void learnFiring(const Neuron<U>* parent, const Synapses& synapses);
    typedef U Value;
    static const Value sigmoid(const Value value);
};


template <typename V>
Neuron<V>::Neuron(const std::string name, Neuron<V>* parent, const unsigned int num, const V coefficientN, const V coefficientA) 
    : name_(name), parent_(parent), child_(NULL), coefficientN_(coefficientN), coefficientA_(coefficientA)
{
    assert(0 < num);

    if (parent) parent->child(this);

    synapses_.resize(num);
    for (unsigned int i = 0; i < num; i++) {
        synapses_[i] = new Synaps<V, Neuron<V> >(*this, (parent ? parent->synapses().size() : 0));
    }
}


template <typename V>
Neuron<V>::~Neuron() 
{
    for (typename Synapses::iterator iter = synapses_.begin(); iter != synapses_.end(); iter++) {
        if (*iter) delete *iter;
    }
    synapses_.clear();
}


template <typename V>
void Neuron<V>::child(Neuron<V>* neural) throw (const unsigned char*)
{
    assert(neural);

    if (child_) throw "Child node has already been specified.";
    child_ = neural;
}


template <typename V>
void Neuron<V>::transmit(const std::vector<V>& outputs) const 
{
    const unsigned int size = synapses().size();
    if (!parent_) {
        for (unsigned int i = 0; i < size; i++)
            synapses()[i]->forwardOutput(outputs[i]);

    } else {
        for (unsigned int i = 0; i < size; i++) {
            V total = Prim<V>::min();
            for (unsigned int j = 0; j < synapses()[i]->forwardWeights().size(); j++) {
                total += synapses()[i]->forwardWeights()[j] * outputs[j];
            }
            total += synapses()[i]->forwardFiring();
            synapses()[i]->forwardOutput(NeuronImpl<V>::sigmoid(total));
        } 
    }

    if (child_) {
        std::vector<V> result(synapses().size());
        for (unsigned int i = 0; i < synapses().size(); i++) {
            result[i] = synapses()[i]->forwardOutput();
        }
        child_->transmit(result);
    }
}


template <typename V>
void Neuron<V>::teach(const std::vector<V>* learns) const
{
    const unsigned int size = synapses().size();
    if (!child_) {
        for (unsigned int i = 0; i < size; i++) {
            synapses()[i]->reverseLearn((learns->at(i) - synapses()[i]->forwardOutput()) * synapses()[i]->forwardOutput() * 
                                        (Prim<V>::max() - synapses()[i]->forwardOutput()));
            NeuronImpl<V>::learnWeight(parent_, synapses_, i);
        }
        NeuronImpl<V>::learnFiring(parent_, synapses_);

    } else if (parent_) { 
        for (unsigned int i = 0; i < size; i++) {
            V total = Prim<V>::min();
            for (unsigned int j = 0; j < child_->synapses().size(); j++) {
                total += child_->synapses()[j]->reverseLearn() * child_->synapses()[j]->forwardWeights()[i];
            }
            synapses()[i]->reverseLearn(synapses()[i]->forwardOutput() * 
                (Prim<V>::max() - synapses()[i]->forwardOutput()) * total);
            NeuronImpl<V>::learnWeight(parent_, synapses_, i);
        }
        NeuronImpl<V>::learnFiring(parent_, synapses_);

    } else { 
        return;

    }

    parent_->teach(NULL);
}  


template <typename V>
void NeuronImpl<V>::learnWeight(const Neuron<V>* parent, const Synapses& synapses, unsigned int index)
{
    assert(parent);

    Synaps< V, Neuron<V> >* target = synapses[index];
    const unsigned int size = target->historyOfReverseWeights().size();

    for (unsigned int i = 0; i < size; i++) {
        const Synaps< V, Neuron<V> >* synaps = parent->synapses()[i];
        target->historyOfReverseWeights()[i] = parent->N() * target->reverseLearn() * synaps->forwardOutput() +
                                               parent->A() * target->historyOfReverseWeights()[i];
        target->forwardWeights(i, target->forwardWeights()[i] + target->historyOfReverseWeights()[i]);
    }
}


template <typename V>
void NeuronImpl<V>::learnFiring(const Neuron<V>* parent, const Synapses& synapses) 
{
    assert(parent);

    const unsigned int size = synapses.size();
    for (unsigned int i = 0; i < size; i++) {
        Synaps< V, Neuron<V> >* synaps = synapses[i];
        synaps->historyOfReverseFiring(parent->N() * synaps->reverseLearn() + 
                                       parent->A() * synaps->historyOfReverseFiring());
        synaps->forwardFiring(synaps->forwardFiring() + synaps->historyOfReverseFiring());
    }
}


template <typename V>
const V NeuronImpl<V>::sigmoid(const V value) 
{
    return (Prim<V>::max() / (Prim<V>::max() + (exp(Prim<V>::min() - value))));
}


/*!
 * @briefi Create an instance of the double type externally provided.
 */
template class Neuron<double>;


/*!
 * @briefi Create an instance of the double type externally provided.
 */
template class Neuron< std::complex<double> >;


}; // end of namespace




