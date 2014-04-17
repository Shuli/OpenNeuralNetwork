/*!
 * @brief  This is a demo of the neural network.
 * @author Hisashi Ikari<ikari@akane.waseda.jp>, Waseda University.
 * @file
 */
#include <iostream>
#include <vector>
#include <cassert>
#include "Network.h"
#include "Neuron.h"

using namespace neuralnet;

typedef Network< double, Neuron<double> > NeuralNet;

/*!
 * @brief The actual processing of the synapse.
 */
class Reasoning
{
public:
    static void processing();
    static void create(const unsigned int number, const unsigned int dimension, 
        std::vector< std::vector<double>* >& values);
    static void remove(const unsigned int number, std::vector< std::vector<double>* >& values);
};


void Reasoning::create(const unsigned int number, const unsigned int dimension, 
    std::vector< std::vector<double>* >& values)
{
    assert(0 < number && 0 < dimension && number == values.size());

    for (unsigned int i = 0; i < number; i++) {
        values[i] = new std::vector<double>(dimension);
    }    
}


void Reasoning::remove(const unsigned int number, std::vector< std::vector<double>* >& values)
{    
    for (unsigned int i = 0; i < number; i++) {
        if (values[i]) delete values[i]; 
    }
    values.clear();
}


void Reasoning::processing()
{
    // Creating the informations of answer and question.
    static const unsigned int EXAMPLES_NUMBER = 8;    

    std::vector< std::vector<double>* > 
        question(EXAMPLES_NUMBER), answer(EXAMPLES_NUMBER);
    
    create(EXAMPLES_NUMBER, 3, question);
    create(EXAMPLES_NUMBER, 1, answer);

    question[0]->at(0) = 1.0; question[0]->at(1) = 1.0; question[0]->at(2) = 1.0;
    question[1]->at(0) = 1.0; question[1]->at(1) = 1.0; question[1]->at(2) = 0.0;
    question[2]->at(0) = 1.0; question[2]->at(1) = 0.0; question[2]->at(2) = 1.0;
    question[3]->at(0) = 1.0; question[3]->at(1) = 0.0; question[3]->at(2) = 0.0;
 
    question[4]->at(0) = 0.0; question[4]->at(1) = 1.0; question[4]->at(2) = 1.0;
    question[5]->at(0) = 0.0; question[5]->at(1) = 1.0; question[5]->at(2) = 0.0;
    question[6]->at(0) = 0.0; question[6]->at(1) = 0.0; question[6]->at(2) = 1.0;
    question[7]->at(0) = 0.0; question[7]->at(1) = 0.0; question[7]->at(2) = 0.0;
       
    answer[0]->at(0) = 1.0; 
    answer[1]->at(0) = 0.0; 
    answer[2]->at(0) = 0.0;
    answer[3]->at(0) = 0.0; 

    answer[4]->at(0) = 1.0; 
    answer[5]->at(0) = 0.0; 
    answer[6]->at(0) = 1.0; 
    answer[7]->at(0) = 1.0; 


    // Creating the hierarchy of the neural network.
    static const unsigned int HIERARCHY_NUMBER = 3; 

    std::vector<std::string> names(HIERARCHY_NUMBER);
    std::vector<unsigned int> size(HIERARCHY_NUMBER);

    names[0] = "input-layer"; size[0] = 3;
    names[1] = "middle-layer"; size[1] = 3;
    names[2] = "output-layer"; size[2] = 1;


    // Learning and identify.
    static const double N = 0.8, A = 0.75, T = 0.08;
    NeuralNet network(HIERARCHY_NUMBER, names, size, N, A);
    network.learning(question, answer, 800, T);

    std::cout << "question\t";
    for (unsigned int i = 0; i < EXAMPLES_NUMBER; i++) {
        std::cout << (i == 0 ? "" : "\t") << "{" 
            << (double)question[i]->at(0) << "," 
            << (double)question[i]->at(1) << "," 
			<< (double)question[i]->at(2) << "}";
    }
    std::cout << std::endl;

    std::cout << "answer\t";
    for (unsigned int i = 0; i < EXAMPLES_NUMBER; i++) {
        const std::vector<double>* result = network.identify(*question[i]);
        const unsigned int resultSize = result->size();
        for (unsigned int j = 0; j < resultSize; j++) {
            std::cout << (i == 0 ? "" : "\t") << "{" << result->at(j) << "}";
        }
        delete result;
    }
    std::cout << std::endl;

}


int main(const int argc, const char** argv) 
{
    Reasoning::processing();    
    return 0;
}


