//
// Created by rishabh on 6/22/22.
//

#include "Node.h"
#include <random>
#include <QDebug>

Node::Node(int size)
{
    weights.resize(size);
    local_grad.resize(size);
    update.fill(0, size);
    prev_update.fill(0, size);

    std::random_device r;
    std::default_random_engine generator{r()};
    std::uniform_real_distribution<float> distribution(0,1);

    for(float & weight : weights)
        weight = distribution(generator);

    bias = distribution(generator);
}

