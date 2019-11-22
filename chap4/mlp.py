import numpy as np

def init_model_hidden() :
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt

    pm_hidden = alloc_param_pair([input_cnt, hidden_cnt])
    pm_output = alloc_param_pair([hidden_cnt, output_cnt])

def alloc_param_pair(shape) :
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])

    return {'w' : weight, 'b' : bias}
    