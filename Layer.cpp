//
// Created by rishabh on 6/22/22.
//

#include "Layer.h"
#include <QDebug>
#include <cmath>

void (*Activation::func[4])(Layer &) = {&linear, &relu, &sigmoid, &softmax};
void (*Activation::deriv[4])(Layer &) = {&linear_deriv, &relu_deriv, &sigmoid_deriv, &softmax_deriv};

void Activation::relu(Layer &layer)
{
    for(int x = 0; x < layer.length(); x++)
        if (layer[x] < 0)
            layer[x] = 0;
}

void Activation::relu_deriv(Layer &layer)
{
    for(int x = 0; x < layer.length(); x++)
    {
        if (layer.at(x) < 0)
            layer[x].gradient = 0;
    }
}

void Activation::sigmoid(Layer &layer)
{
    for(int x = 0; x < layer.length(); x++)
        layer[x] = 1/(1 + exp((layer[x] + 1e-15) * -1));
}

void Activation::sigmoid_deriv(Layer &layer)
{
    for(int x = 0; x < layer.length(); x++)
        layer[x].gradient *= (1/(1 + exp((layer[x] + 1e-15) * -1))) * (1 - (1/(1 + exp((layer[x] + 1e-15) * -1))));
}

void Activation::softmax(Layer &layer)
{
    float max = -INFINITY;
    for(int x = 0; x < layer.length(); x++)
        if(layer[x] > max)
            max = layer[x].value;

    float sum = 0;
    for(int x = 0; x < layer.length(); x++)
        sum += exp(layer[x].value - max);

    float offset = max + log(sum);
    for(int x = 0; x < layer.length(); x++)
        layer[x] = exp(layer[x].value - offset);
}


void Layer::operator*(Layer prev_layer)
{
    for(int x = 0; x < this->length(); x++)
    {
        for (int y = 0; y < prev_layer.length(); y++)
        {
            this->data()[x] += (prev_layer[y] * this->at(x).weights[y]);
            this->data()[x].local_grad[y] = prev_layer[y].value;
        }
        this->data()[x] += this->at(x).bias;
        this->data()[x].local_grad_bias = 1;
    }
    Activation::func[type](*this);
}

void Layer::operator/(Layer next_layer)
{
    for(int x = 0; x < this->length(); x++)
    {
        this->data()[x].gradient = 0;
        for (int y = 0; y < next_layer.length(); y++)
        {
            this->data()[x].gradient += next_layer[y].weights[x] * next_layer[y].gradient;
        }

    }
    Activation::deriv[type](*this);
}

void Layer::operator=(QVector<float> input)
{
    for(int x = 0; x < this->length(); x++)
        this->data()[x] = input[x];
}
