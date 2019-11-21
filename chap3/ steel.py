import numpy as np

def steel_exec(epoch_count = 10, mb_size = 10, report = 1) :
    load_steel_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_steel_dataset() :
    with opne('../../data/chap3/faults.csv') as csvfile :
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader :
            rows.append(row)

    global data, input_cnt, output_cnt
    data = np.asarray(rows, dtype = 'float32')
    input_cnt, output_cnt = 27, 7

def forward_postproc(output, y) :
    entropy = softmax_cross_with_logits(y, output)
    loss = np.mean(entropy)
    return loss, [y, output, entropy]

def backprop_postproc(G_loss, aux) :
    y, output, entropy = aux

    g_loss_entropy = 1.0 / np.prod(entropy.shape)
    g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

    G_entropy = g_entropy_output * G_loss
    G_output = g_entropy_output * G_entropy

    return G_output

def eval_accuracy(output, y) :
    estimate = np.argmax(output, axis=1)
    answer = np.argmax(y, axis=1)
    correct = np.equal(estimate, answer)

    return np.mean(correct)

def softmax(x) :
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()

    return probs

def softmax_derv(x, y) :
    mb_size, nom_size = x.shape
    derv = np.ndarray([mb_size, nom_size, nom_size])
    for n in range(mb_size) :
        for i in range(nom_size) :
            for j in range(nom_size) :
                derv[n, i, j] = -y[n, i] * y[n,j]
            derv[n, i, i] += y[n, i]
    
    return derv