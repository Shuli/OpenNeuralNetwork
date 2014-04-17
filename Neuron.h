/*!
 * @brief  This is a neural network.
 * @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.
 * @file
 */
#ifndef NEURAL_NEURAL_H_INCLUDED
#define NEURAL_NEURAL_H_INCLUDED

#include <string>
#include "Synaps.h"

namespace neuralnet {

/*!
 * @brief Definition of "LAYER" of the neuron.
 */
template <typename V>
class Neuron
{
public:
    Neuron(const std::string name, Neuron<V>* parent, const unsigned int num, 
        const V coefficientN, const V coefficientA);
    virtual ~Neuron();

    void transmit(const std::vector<V>& outputs) const;
    void teach(const std::vector<V>* learns) const;    
    void child(Neuron<V>* neural) throw (const unsigned char*);

    typedef std::vector< Synaps< V, Neuron<V> >* > Synapses;
    const Synapses& synapses() const { return synapses_; }
    const std::string& name() const { return name_; }

    const V N() const { return coefficientN_; }
    const V A() const { return coefficientA_; }

protected:
    const std::string name_;
    const Neuron<V>* parent_;
    Neuron<V>* child_;
    Synapses synapses_;    

    const V coefficientN_;
    const V coefficientA_;
};

}; //end of namespace

#endif


