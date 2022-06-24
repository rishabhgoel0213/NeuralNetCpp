//
// Created by rishabh on 6/22/22.
//

#ifndef FINALNNCPP_LAYER_H
#define FINALNNCPP_LAYER_H
#include "Node.h"
class Layer;



typedef enum
{
    LINEAR = 0,
    RELU = 1,
    SIGMOID = 2,
    SOFTMAX = 3
} TYPE;

struct Activation
{
    static void linear(Layer &layer){}
    static void linear_deriv(Layer &layer){}

    static void relu(Layer &layer);
    static void relu_deriv(Layer &layer);

    static void sigmoid(Layer &layer);
    static void sigmoid_deriv(Layer &layer);

    static void softmax(Layer &layer);
    static void softmax_deriv(Layer & layer){}

    static void (*func[4])(Layer &);
    static void (*deriv[4])(Layer &);
};


struct Layer: public QVector<Node>
{
    explicit Layer(int size, TYPE t = LINEAR): QVector<Node>(size, 0){type = t;}

    void operator*(const Layer &prev_layer);
    void operator/(const Layer &next_layer);

    void operator=(QVector<float> input);

    TYPE type;
};




#endif //FINALNNCPP_LAYER_H
