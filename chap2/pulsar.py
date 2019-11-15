import numpy as np

def pulsar_exec(epoch_count = 10, mb_size = 10, report = 1):
    load_pulsar_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_pulsar_dataset():
    with open("../../data/chap02/pulsar_starts.csv") as csvfile :
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []

        for row in csvreader :
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1
    data = np.asarray(rows, dtype = 'float32')


def forward_postproc(output, y) :
    entropy = sigmoid_cross_entropy_with_logits(y, output)
    loss = np.mean(entropy)
    return loss, [y, output, entropy]

def backprop_postproc(G_loss, aux) :
    y, output, entropy = aux

    g_loss_entropy = 1.0 / np.prod(entropy.shape)
    g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

    G_entropy = g_loss_entropy * G_loss
    G_output = g_entropy_output * G_entropy

    return G_output

def eval_accuracy(output, y) :
    estimate = np.greater(output, 0)
    answer = np.greater(y, 0.5)
    correct = np.equal(estimate, answer)

    return np.mean(correct)

def relu(x) :
    return np.maximum(x, 0)

def sigmoid(x) :
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))

