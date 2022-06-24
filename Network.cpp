//
// Created by rishabh on 6/22/22.
//

#include "Network.h"
#include <cmath>
#include <QDebug>

float (*Loss::func[3])(const Layer &, const QVector<float> &) = {&cross_entropy, &cross_entropy, &mean_squared};
void (*Loss::deriv[3])(Layer &, const QVector<float> &) = {&cross_deriv, &cross_softmax_deriv, &squared_deriv};

Network::Network(const QVector<Layer> &net, LOSS_TYPE l): QVector<Layer>(net)
{
    loss_type = l;
    for(int x = 0; x < net[0].length(); x++)
        this->data()[0][x] = Node(0);

    for(int x = 1; x < net.length(); x++)
        for(int y = 0; y < net[x].length(); y++)
                this->data()[x][y] = Node(this->at(x - 1).length());
}

void Network::forward_prop(const QVector<float>& input_values)
{
    this->data()[0] = input_values;

    for(int x = 1; x < this->length(); x++)
        this->data()[x] * this->data()[x - 1];

}

float Network::calc_loss(const QVector<float> &correct_values)
{
    QVector<float> output_values;
    float output_hold = 0;
    int output_value;
    for(int x = 0; x < this->constData()[this->length() - 1].length(); x++)
        if(output_hold < this->constData()[this->length() - 1][x].value)
        {
            output_hold = this->constData()[this->length() - 1][x].value;
            output_value = x;
        }

    int input_value;
    for(int x = 0; x < correct_values.length(); x++)
        if(correct_values[x] == 1)
            input_value = x;

    float loss = Loss::func[loss_type](this->constData()[this->length() - 1], correct_values);

    qDebug() << loss << ", " << input_value << ", " << output_value;

    return loss;
}


void Network::backward_prop(const QVector<float> &correct_values)
{
    Loss::deriv[loss_type](this->data()[this->length() - 1], correct_values);
    Activation::deriv[this->data()[this->length() - 1].type](this->data()[this->length() - 1]);

    for(int x = this->length() - 2; x > 0; x--)
        this->data()[x] / this->data()[x + 1];
}

void Network::calc_grad(float momentum)
{
    for(int x = 0; x < this->length(); x++)
        for(int y = 0; y < this->constData()[x].length(); y++)
        {
            float hold = this->data()[x][y].update_bias;
            this->data()[x][y].update_bias = (this->data()[x][y].local_grad_bias * this->data()[x][y].gradient) + (momentum * this->data()[x][y].prev_update_bias);
            this->data()[x][y].prev_update_bias = hold;
            for(int z = 0; z < this->constData()[x][y].weights.length(); z++)
            {
                float hold_w = this->data()[x][y].update[z];
                this->data()[x][y].update[z] = (this->data()[x][y].local_grad[z] * this->data()[x][y].gradient) +
                                               (momentum * this->data()[x][y].prev_update[z]);
                this->data()[x][y].prev_update[z] = hold_w;
            }
        }
}

void Network::update(float lr, float decay, int epoch)
{
    lr = (1/(1+decay*epoch)) * lr;
    for(int x = 0; x < this->length(); x++)
        for(int y = 0; y < this->at(x).length(); y++)
        {
//            this->data()[x][y].bias -= this->constData()[x][y].gradient * this->constData()[x][y].local_grad_bias * lr;
            this->data()[x][y].bias -= this->data()[x][y].update_bias * lr;
//            qDebug() << (this->constData()[x][y].update_bias == this->constData()[x][y].gradient * this->constData()[x][y].local_grad_bias);
//            this->data()[x][y].update_bias = 0;
            for (int z = 0; z < this->at(x).at(y).weights.length(); z++)
            {
//                this->data()[x][y].weights[z] -=  this->constData()[x][y].gradient * this->constData()[x][y].local_grad[z] * lr;
                this->data()[x][y].weights[z] -=  this->data()[x][y].update[z] * lr;
//                this->data()[x][y].update[z] = 0;
            }
        }
}

void Network::clear_nodes()
{
    for(int x = 0; x < this->length(); x++)
        for(int y = 0; y < this->at(x).length(); y++)
            this->data()[x][y] = 0;
}


float Loss::cross_entropy(const Layer &predicted_values, const QVector<float> &correct_values)
{
    float loss = 0;
    for(int x = 0; x < predicted_values.length(); x++)
            loss += correct_values[x] * log(predicted_values[x].value + 1e-15);
//    loss = loss / predicted_values.length();
    return loss;
}

void Loss::cross_deriv(Layer &predicted_values, const QVector<float> &correct_values)
{
    for(int x = 0; x < predicted_values.length(); x++)
        predicted_values[x].gradient = -1 * (correct_values[x] / (predicted_values[x].value + 1e-15)) + ((1 - correct_values[x]) / (1 - (predicted_values[x].value + 1e-15)));
}

float Loss::mean_squared(const Layer &predicted_values, const QVector<float> &correct_values)
{
    float loss = 0;
    for(int x = 0; x < predicted_values.length(); x++)
        loss += pow(correct_values[x] - predicted_values[x].value, 2);
    loss = loss / predicted_values.length();
    return loss;
}

void Loss::squared_deriv(Layer &predicted_values, const QVector<float> &correct_values)
{
    for(int x = 0; x < predicted_values.length(); x++)
        predicted_values[x].gradient = -2 * (correct_values[x] - predicted_values[x].value);
}

void Loss::cross_softmax_deriv(Layer &predicted_values, const QVector<float> &correct_values)
{
    for(int x = 0; x < predicted_values.length(); x++)
        predicted_values[x].gradient = correct_values[x] - predicted_values[x].value;
}
