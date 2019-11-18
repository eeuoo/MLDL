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

