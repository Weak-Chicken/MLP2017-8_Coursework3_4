import numpy as np
import matplotlib.pyplot as plt

input_dim = 518
target_dim = 4858
epochs = 100

layers = [2, 3, 4]
neurons = [200, 400, 600, 800]


def draw_default(folderPath):
    ACC = np.zeros(epochs)
    ERR = np.zeros(epochs)
    predict_ACC = np.zeros(epochs)
    predict_ERR = np.zeros(epochs)

    for layer in layers:
        for neuron_num in neurons:
            ERR = np.loadtxt('Results/{}/err_L{}_N{}.csv'.format(folderPath, layer, neuron_num), delimiter=",")
            ACC = np.loadtxt('Results/{}/acc_L{}_N{}.csv'.format(folderPath, layer, neuron_num), delimiter=",")
            predict_ERR = np.loadtxt('Results/{}/pre_err_L{}_N{}.csv'.format(folderPath, layer, neuron_num), delimiter=",")
            predict_ACC = np.loadtxt('Results/{}/pre_acc_L{}_N{}.csv'.format(folderPath, layer, neuron_num), delimiter=",")

            plt.plot(np.arange(epochs), ERR, 'r')
            plt.plot(np.arange(epochs), predict_ERR, 'g')
            plt.xlabel('Epochs')
            plt.ylabel('ERR')
            plt.title('Deep neural network performance')
            plt.legend(['Train', 'Vaild'], loc='upper right')
            plt.savefig('Figures/{}/ERR_plots_L{}_N{}.pdf'.format(folderPath, layer, neuron_num))
            plt.close()

            plt.plot(np.arange(epochs), ACC, 'r')
            plt.plot(np.arange(epochs), predict_ACC, 'g')
            plt.xlabel('Epochs')
            plt.ylabel('ACC')
            plt.title('Deep neural network performance')
            plt.legend(['Train', 'Vaild'], loc='upper right')
            plt.savefig('Figures/{}/ACC_plots_L{}_N{}.pdf'.format(folderPath, layer, neuron_num))
            plt.close()


def draw_overall(folderPath):
    ERR = np.zeros([len(layers), len(neurons), epochs])
    ACC = np.zeros([len(layers), len(neurons), epochs])
    predict_ERR = np.zeros([len(layers), len(neurons), epochs])
    predict_ACC = np.zeros([len(layers), len(neurons), epochs])

    for layer in range(len(layers)):
        for neuron_num in range(len(neurons)):

            err = np.loadtxt('Results/{}/err_L{}_N{}.csv'.format(folderPath, layers[layer], neurons[neuron_num]), delimiter=",")
            acc = np.loadtxt('Results/{}/acc_L{}_N{}.csv'.format(folderPath, layers[layer], neurons[neuron_num]), delimiter=",")
            predict_err = np.loadtxt('Results/{}/pre_err_L{}_N{}.csv'.format(folderPath, layers[layer], neurons[neuron_num]), delimiter=",")
            predict_acc = np.loadtxt('Results/{}/pre_acc_L{}_N{}.csv'.format(folderPath, layers[layer], neurons[neuron_num]), delimiter=",")

            ERR[layer, neuron_num] = err
            ACC[layer, neuron_num] = acc
            predict_ERR[layer, neuron_num] = predict_err
            predict_ACC[layer, neuron_num] = predict_acc

    for layer in range(len(layers)):
        for neuron_num in range(len(neurons)):
            plt.plot(np.arange(epochs), ERR[layer, neuron_num])
    plt.xlabel('Epochs')
    plt.ylabel('Train_error')
    plt.title('Deep neural network performance')
    plt.legend(['2-200', '2-400', '2-600', '2-800',
                '3-200', '3-400', '3-600', '3-800',
                '4-200', '4-400', '4-600', '4-800'], loc='upper right')
    plt.savefig('Figures/{}/Train_error_plots_L{}_N{}.pdf'.format(folderPath,layer, neuron_num))
    plt.close()

    for layer in range(len(layers)):
        for neuron_num in range(len(neurons)):
            plt.plot(np.arange(epochs), ACC[layer, neuron_num])
    plt.xlabel('Epochs')
    plt.ylabel('Train_accuracy')
    plt.title('Deep neural network performance')
    plt.legend(['2-200', '2-400', '2-600', '2-800',
                '3-200', '3-400', '3-600', '3-800',
                '4-200', '4-400', '4-600', '4-800'], loc='upper right')
    plt.savefig('Figures/{}/Train_accuracy_plots_L{}_N{}.pdf'.format(folderPath,layer, neuron_num))
    plt.close()

    for layer in range(len(layers)):
        for neuron_num in range(len(neurons)):
            plt.plot(np.arange(epochs), predict_ERR[layer, neuron_num])
    plt.xlabel('Epochs')
    plt.ylabel('Valid_error')
    plt.title('Deep neural network performance')
    plt.legend(['2-200', '2-400', '2-600', '2-800',
                '3-200', '3-400', '3-600', '3-800',
                '4-200', '4-400', '4-600', '4-800'], loc='upper right')
    plt.savefig('Figures/{}/Valid_error_plots_L{}_N{}.pdf'.format(folderPath,layer, neuron_num))
    plt.close()

    for layer in range(len(layers)):
        for neuron_num in range(len(neurons)):
            plt.plot(np.arange(epochs), predict_ACC[layer, neuron_num])
    plt.xlabel('Epochs')
    plt.ylabel('Valid_accuracy')
    plt.title('Deep neural network performance')
    plt.legend(['2-200', '2-400', '2-600', '2-800',
                '3-200', '3-400', '3-600', '3-800',
                '4-200', '4-400', '4-600', '4-800'], loc='upper right')
    plt.savefig('Figures/{}/Valid_accuracy_plots_L{}_N{}.pdf'.format(folderPath,layer, neuron_num))
    plt.close()


if __name__ == '__main__':
    # draw_default('Abandoned-10-LR0.002')
    # draw_default('ALL-LR0.000002')
    # draw_default('Reduced-LR0.000002')
    # draw_overall('Abandoned-10-LR0.002')
    # draw_overall('ALL-LR0.000002')
    # draw_overall('Reduced-LR0.000002')
    draw_default('standard')
