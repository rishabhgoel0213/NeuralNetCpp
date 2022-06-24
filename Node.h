//
// Created by rishabh on 6/22/22.
//

#ifndef FINALNNCPP_NODE_H
#define FINALNNCPP_NODE_H

#include <QVector>


class Node
{
public:
    float value = 0;
    QVector<float> weights;
    float bias = 0;

    QVector<float> local_grad;
    float local_grad_bias = 0;
    float gradient = 0;

    QVector<float> update;
    float update_bias = 0;

    QVector<float> prev_update;
    float prev_update_bias = 0;

    Node(int size);

    void operator=(const float &v) {value = v;}
    void operator+=(const float &v) {value += v;};

    float operator+(const float &v) const {return value + v;}
    float operator*(const float &v) const {return value * v;}

    bool operator>(const float v) const{return this->value > v;}
    bool operator<(const float v) const{return this->value < v;}
};


#endif //FINALNNCPP_NODE_H
