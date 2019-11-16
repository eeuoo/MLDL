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
