//
// Created by rishabh on 6/22/22.
//

#ifndef FINALNNCPP_NETWORK_H
#define FINALNNCPP_NETWORK_H

#include <utility>

#include "Layer.h"

typedef enum
{
    CROSS_ENTROPY = 0,
    CROSS_SOFTMAX = 1,
    MEAN_SQUARED = 2
} LOSS_TYPE;

struct Loss
{
    static float (*func[3])(const Layer &, const QVector<float> &);
    static void (*deriv[3])(Layer &, const QVector<float> &);

    static float cross_entropy(const Layer &predicted_values, const QVector<float>& correct_values);
    static void cross_softmax_deriv(Layer &predicted_values, const QVector<float>& correct_values);
    static void cross_deriv(Layer &predicted_values, const QVector<float>& correct_values);

    static float mean_squared(const Layer &predicted_values, const QVector<float>& correct_values);
    static void squared_deriv(Layer &predicted_values, const QVector<float>& correct_values);
};

class Network: QVector<Layer>
{
public:
    Network(const QVector<Layer> &net, LOSS_TYPE l);

    void forward_prop(const QVector<float>& input_values);
    float calc_loss(const QVector<float>& correct_values);
    void backward_prop(const QVector<float>& correct_values);
    void calc_grad(float momentum);
    void update(float lr, float decay = 0, int epoch = 0);
    void clear_nodes();

private:
    LOSS_TYPE loss_type;
    Loss loss_func;
};


#endif //FINALNNCPP_NETWORK_H
