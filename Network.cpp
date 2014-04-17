/*!
 * @brief  Entrance(Facade) of the neural network. 
 * @author Hisashi<ikari@akane.waseda.jp>, Waseda University. 
 * @file
 */
#define _SHOW_CALC_PROCESS

#ifdef _SHOW_CALC_PROCESS
#include <iostream>
#endif
#include <vector>
#include <complex>
#include <cassert>
#include <stdlib.h>

#include "Synaps.h"
#include "Neuron.h"
#include "Network.h"

namespace neuralnet {

template <typename V, class R>
class NetworkImpl
{
public:
    NetworkImpl(const unsigned int hierarchySize, 
        const std::vector<std::string>& hierarchyNames, const std::vector<unsigned int>& neuronSize,
        const V coefficientN, const V coefficientA);
    ~NetworkImpl();

    void createNeurons();
    void removeNeurons();

    static const double difference(const V& lefthand, const V& righthand);

public:
    typedef std::vector<R*> Neurons;
    Neurons neurons_;

    const unsigned int hierarchySize_;
    const std::vector<std::string>& hierarchyNames_;
    const std::vector<unsigned int>& neuronSize_;
    const V coefficientN_;
    const V coefficientA_; 
};


template <typename V, class R>
Network<V, R>::Network(const unsigned int hierarchySize, 
    const std::vector<std::string>& hierarchyNames, const std::vector<unsigned int>& neuronSize, 
    const V coefficientN, const V coefficientA) 
{
    impl_ = new NetworkImpl<V, R>(hierarchySize, hierarchyNames, neuronSize, coefficientN, coefficientA);
}


template <typename V, class R>
Network<V, R>::~Network()
{
    if (impl_) delete impl_;
}


template <typename V, class R>
NetworkImpl<V, R>::NetworkImpl(const unsigned int hierarchySize, 
    const std::vector<std::string>& hierarchyNames, const std::vector<unsigned int>& neuronSize,
    const V coefficientN, const V coefficientA)
        : hierarchySize_(hierarchySize), hierarchyNames_(hierarchyNames), neuronSize_(neuronSize),
            coefficientN_(coefficientN), coefficientA_(coefficientA) {}


template <typename V, class R>
NetworkImpl<V, R>::~NetworkImpl()
{
    removeNeurons();
}


template <typename V, class R>
void NetworkImpl<V, R>::createNeurons()
{
    assert(hierarchySize_ == hierarchyNames_.size() && 0 < hierarchySize_);

    neurons_.resize(hierarchySize_);
    for (unsigned int i = 0; i < hierarchySize_; i++) {
        neurons_[i] = new R(hierarchyNames_[i], (i == 0 ? NULL : neurons_[i - 1]), 
            neuronSize_[i], coefficientN_, coefficientA_);
    }    
}


template <typename V, class R>
void NetworkImpl<V, R>::removeNeurons()
{
    for (typename Neurons::const_iterator iter = neurons_.begin(); iter != neurons_.end(); iter++) {
        if (*iter) delete *iter;
    }
    neurons_.clear();
}


template <typename V, class R>
void Network<V, R>::learning(const std::vector< std::vector<V>* >& question, const std::vector< std::vector<V>* >& teach, 
    const unsigned int maxTrialNum, const double stopThreshold)
{
    assert((question.size() == teach.size()) && 0 < question.size() && 0 < teach.size());

    unsigned int number = maxTrialNum;
    while (number >= maxTrialNum) {
        number = 0;

        srand((unsigned)time(NULL));

        impl_->removeNeurons();
        impl_->createNeurons();

        const R* inputLayer = impl_->neurons_.front();
        const R* outputLayer = impl_->neurons_.back();

        const unsigned int dataSize = question.size();
        const unsigned int outputSize = outputLayer->synapses().size();

        double error = stopThreshold;        
        while (error >= stopThreshold) {
            for (unsigned int i = 0; i < dataSize; i++) {
                inputLayer->transmit(*question[i]);
                outputLayer->teach(teach[i]);
            }
            error = 0.0;
            number++;
#ifdef _SHOW_CALC_PROCESS
            std::cout << number << "\t";
#endif
            for (unsigned int i = 0; i < dataSize; i++) {
                inputLayer->transmit(*(question[i]));
                for (unsigned int j = 0; j < outputSize; j++) {
                    const V difference = (teach[i]->at(j) - 
                        outputLayer->synapses()[j]->forwardOutput()); 
                    error += NetworkImpl<V, R>::difference(difference, difference);
                }   
#ifdef _SHOW_CALC_PROCESS
                for (unsigned int j = 0; j < outputSize; j++) {
                    std::cout << outputLayer->synapses()[j]->forwardOutput() << "\t";
                } 
#endif
            }
            error *= 0.5;
#ifdef _SHOW_CALC_PROCESS
            std::cout << error << std::endl;
#endif
            if (number >= maxTrialNum) {
                break;
            }
        }

    }
}


template <typename V, class R>
const double NetworkImpl<V, R>::difference(const V& lefthand, const V& righthand)
{
    return lefthand * righthand;
} 


typedef std::complex<double> Complex;
template <>
const double NetworkImpl< Complex, Neuron<Complex> >::difference(const Complex& lefthand, const Complex& righthand)
{
    // This is not a correct answer. Not implemented yet.
    return 0.0;
}


template <typename V, class R>
const std::vector<V>* Network<V, R>::identify(const std::vector<V>& question)
{
    assert(0 < question.size() && 0 < impl_->neurons_.size());

    const R* inputLayer = impl_->neurons_.front();
    const R* outputLayer = impl_->neurons_.back();
    const unsigned int outputSize = outputLayer->synapses().size();

    inputLayer->transmit(question);

    std::vector<V>* result = new std::vector<V>();
    result->resize(outputSize);
    for (unsigned int i = 0; i < outputSize; i++) {
        result->at(i) = outputLayer->synapses()[i]->forwardOutput();
    }

    return result;
}


/*!
 * @brief Create an instance of the double type externally provided.
 */
template class Network< double, Neuron<double> >;
template class NetworkImpl < double, Neuron<double> >;


/*!
 * @brief Create an instance of the double type externally provided.
 */
template class Network< Complex, Neuron<Complex> >;
template class NetworkImpl < Complex, Neuron<Complex> >;


}; // end of namespace




