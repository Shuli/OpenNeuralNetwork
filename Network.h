/*!
 * @brief  Entrance(Facade) of the neural network. 
 * @author Hisashi<ikari@akane.waseda.jp>, Waseda University. 
 * @file
 */
#ifndef NEURAL_NETWORK_H_INCLUDED
#define NEURAL_NETWORK_H_INCLUDED

#include <string>
#include <vector>

namespace neuralnet {

template <typename V, class R>
class NetworkImpl;

/*!
 * @brief Entrance(Facade) of the neural network.
 */
template <typename V, class R>
class Network
{
public:
    Network(const unsigned int hierarchySize, 
        const std::vector<std::string>& neuronNames, const std::vector<unsigned int>& neuronSize,
        const V coefficientN, const V coefficientA);
    ~Network();

    void learning(const std::vector< std::vector<V>* >& question, const std::vector< std::vector<V>* >& teach,
        const unsigned int maxTrialNum, const double stopThreshold);
    const std::vector<V>* identify(const std::vector<V>& question);

protected:
    NetworkImpl<V, R>* impl_;

};

}; //end of namespace

#endif


