#include <iostream>
#include <QFile>
#include "Network.h"
#include <QDebug>
#include <random>
#include <QProcess>
#include "LayerClasses/Dense.h"

float scale(float value, float min, float max)
{
    return ((value - min)/(max - min));
}

QVector<QVector<float>> transponse(QVector<QVector<float>> input)
{
    QVector<QVector<float>> output(input[0].length(), QVector<float>(input.length()));
    for(int x = 0; x < input[0].length(); x++)
    {
        for (int y = 0; y < input.length(); y++)
        {
            output[x][y] = input[y][x];
        }
    }

    return output;
}

QVector<QVector<float>> read_csv(QString file_string, int cols)
{
    QFile file(file_string);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << file.errorString();
        return QVector<QVector<float>>();
    }

    QVector<QVector<float>> wordList;
//    QStringList wordList;
    wordList.resize(cols);
    while (!file.atEnd())
    {
        QByteArray line = file.readLine();
        for(int x = 0; x < wordList.length(); x++)
        {
            wordList[x].append(line.simplified().split(' ').at(x).toFloat());
        }

    }
    return wordList;
}

void plot_csv(QVector<float> losses)
{
    freopen("output.csv", "w", stdout);
    for (int x = 0; x < losses.length(); x++)
    {
        std::cout << losses[x] << std::endl;
    }
    QStringList args{"../plot_csv.py"};
    QProcess p;
    p.start("python3", args);
    p.waitForFinished();
}

void test(Network net, QString file)
{
    QFile test(file);
    QVector<float> losses;
    QVector<float> data(28 * 28 + 1);
    QVector<float> values(10, 0);
    QByteArray line;

    if (!test.open(QIODevice::ReadOnly))
        qDebug() << test.errorString();

    while(!test.atEnd())
    {
        values.fill(0);
        line = test.readLine();
        for (int x = 0; x < data.length(); x++)
            data[x] = line.split(',').at(x).toFloat();

        int output_value = (int) data.takeFirst();

        for (float & x : data)
            x /= 255;
        values[output_value] = 1;

        net.forward_prop(data);
        losses.append(net.calc_loss(values));
        net.clear_nodes();
    }
    plot_csv(losses);
}

void train(Network &net, const QString& file, float lr = 0.001, float momentum = 0)
{
    QFile train(file);
    QVector<float> losses;

    QByteArray line;

    if (!train.open(QIODevice::ReadOnly))
        qDebug() << train.errorString();

    while(!train.atEnd())
    {
        QVector<float> data(28 * 28 + 1);
        QVector<float> values(10, 0);
        line = train.readLine();
        for (int x = 0; x < data.length(); x++)
            data[x] = line.split(',').at(x).toFloat();

        int output_value = (int) data.takeFirst();

        for (float & x : data)
            x /= 255;
        values[output_value] = 1;

        net.forward_prop(data);
        losses.append(net.calc_loss(values));
        net.backward_prop(values);
        net.calc_grad(0.1);
        net.update(0.0005);
        net.clear_nodes();
    }
    plot_csv(losses);
}

int main()
{
    Network net(
    QVector<Layer>
    {
             Layer(28 * 28),
             Layer(256, RELU),
             Layer(256, RELU),
             Layer(10, SIGMOID)
    }, MEAN_SQUARED);

    train(net, "mnist_train.csv", 0.0005, 0.1);
    test(net, "mnist_test.csv");

//    QVector<QVector<float>> data = read_csv("auto-mpg.data", 8);
//
//
////    float max_output = *std::max_element(data[0].constBegin(), data[0].constEnd());
//    qDebug() << data;
//
//    //Data Scaling
//    for(int x = 0; x < data.length(); x++)
//    {
//        float min = *std::min_element(data[x].constBegin(), data[x].constEnd());;
//        float max = *std::max_element(data[x].constBegin(), data[x].constEnd());;
//        for(int y = 0; y < data[x].length(); y++)
//        {
//            data[x][y] = scale(data[x][y], min, max);
//        }
//    }
//    QVector<QVector<float>> values(data[0].length());
//    for(int x = 0; x < values.length(); x++)
//        values[x].append(data[0].takeFirst());
//    qDebug() << values;
//    data.pop_front();
//
//    data = transponse(data);
//    QVector<QVector<float>> data_shuffle = data;
//    QVector<QVector<float>> values_shuffle = values;


//    QFile train("mnist_train.csv");
//    QFile test("mnist_test.csv");
//
//    if (!train.open(QIODevice::ReadOnly))
//    {
//        qDebug() << train.errorString();
//        return -1;
//    }
//    if (!test.open(QIODevice::ReadOnly))
//    {
//        qDebug() << test.errorString();
//        return -1;
//    }
//
//    int epoch = 0;
//    QVector<float> loss_history;
//    QVector<float> test_history;
//
//    for(int y = 0; y < 10000; y++)
//    {
//        while(!train.atEnd())
//        {
//            QVector<float> data(28 * 28 + 1);
//            QVector<float> values(10);
//            QByteArray train_line;
//
//            QVector<float> test_data(28 * 28 + 1);
//            QVector<float> test_values(10);
//            QByteArray test_line;
//
//            values.fill(0);
//            test_values.fill(0);
//
//            train_line = train.readLine();
//            for (int x = 0; x < data.length(); x++)
//                data[x] = train_line.split(',').at(x).toFloat();
//
//            int output_value = (int) data.takeFirst();
//
//            for (float & x : data)
//                x /= 255;
//            values[output_value] = 1;
//
//            test_line = test.readLine();
//            for (int x = 0; x < data.length(); x++)
//                test_data[x] = test_line.split(',').at(x).toFloat();
//
//            int test_output_value = (int) test_data.takeFirst();
//
//            for (float & x : test_data)
//                x /= 255;
//            test_values[test_output_value] = 1;
//
//            net.forward_prop(test_data);
//            test_history.append(net.calc_loss(test_values));
//            net.clear_nodes();
//
//            net.forward_prop(data);
//            loss_history.append(net.calc_loss(values));
//            net.backward_prop(values);
//            net.calc_grad(0.1);
//            net.update(0.0005);
//            net.clear_nodes();
//
//            epoch++;
//        }
//    }
    return 0;
}
