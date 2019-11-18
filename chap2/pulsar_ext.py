import pulsar
import pandas as pd
import numpy as np

def pulsar_exec(epoch_count = 10, mb_size = 10, report = 1, adjust_ratio = False) :
    load_pulsar_dataset(adjust_ratio)
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_pulsar_dataset(adjust_ratio) :
    pulsars, stars = [], []
    with open('../../data/chap2/pulsar_stars.csv') as csvfile :
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader :
            if row[8] == '1' :
                pulsars.append(row)
            else :
                stars.append(row)
            
    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1

    star_cnt, pulsar_cnt = len(stars), len(pulsars)

    if adjust_ratio :
        data = np.zeros([2*star_cnt, 9])
        data[0 : star_cnt, : ] = np.asarray(starts, dtype = 'float32')
        for n in range(star_cnt) :
            data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype =' float32')

    else :
        data = np.zeros([star_cnt + pulsar_cnt, 9])
        data[0 : star_cnt,  : ] = np.asarray(stars, dtype = 'float32')
        data[star_cnt,  : ] = np.asarray(pulsars, dtype = 'float32')

def eval_accuracy(output, y) :
    est_yes = np.greater(output, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 2 * safe_div(recall * precision, recall + precision)

    return [accuracy, precision, recall, f1]

def save_div(p, q) :
    p, q = float(p), float(q)
    
    if np.abs(q) < 1.0e-20 :
        return np.sign(p)
    
    return p / q

def train_and_test(epoch_count, mb_size, report) :
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count) :
        losses = []

        for n in range(step_count) :
            train_x, train_y = get_train_data(mb_size, n)
            loss, _ = run_train(train_x, train_y)
            losses.append(loss)

        if report > 0 and (epoch + 1) % report == 0 :
            acc = run_test(test_x, test_y)
            acc_str = ','.join(['%5.3f']*4) % tuple(acc)
            print('Epoch {} : loss = {:5.3f}, result = {}'.format(epoch + 1, np.mean(losses), acc_str))

    acc = run_test(test_x, test_y)
    acc_str = ','.join(['%5.3f']*4) % tuple(acc)
    print('\nFinal Test : final result = {}'.format(acc_str))
        