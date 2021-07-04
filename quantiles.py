from datasketches import kll_ints_sketch
from sklearn.model_selection import train_test_split
from bisect import bisect_left
import time
import gc
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict

data = pd.read_csv('./exponential.csv', header = None)

train, test = train_test_split(data, test_size=0.2)
train = train.sort_index()
test = test.sort_index()
train_keys = train.ix[:,0].tolist()
train_blocks = train.ix[:, 1].tolist()
test_keys = test.ix[:,0].tolist()
test_blocks = test.ix[:, 1].tolist()

number_of_blocks_in_memory = max(data.ix[0:, 1])

def main():
    best_k = 0
    best_cost = -1
    for k in [8, 16, 32, 64, 128,256,512,1024,2048]:
        print("for k = {}:".format(k))
        train_start = time.time()
        (mapping, kll) = train(k)
        train_end = time.time()
        print("Train time: {} ms".format(1000*(train_end-train_start)))
        test_start = time.time()
        (hist, total) = test(mapping)
        test_end = time.time()
        print("Search time: {} ms".format(1000*(test_end-test_start)))
        cost = calculate_cost(hist,total)
        if best_cost == -1 or cost < best_cost:
            best_cost = cost
            best_k = k
        print("Memory size: {} bytes".format(get_model_size_from_details(kll)))
        print("Mean cost: {}".format(cost))
        # if we want to print model size:
        # print("details:")
        # print(kll.to_string())
        print("------------------------------")
        del mapping
        del kll
        gc.collect()
    print("best k is {} with cost {}".format(best_k, best_cost))
    original_time = test_all_without_indexes()
    print("Search time without index: {} ms".format(original_time))

def test_all_without_indexes():
    keys_without_duplicates = list(dict.fromkeys(test_keys))
    start_time = time.time()
    for key in keys_without_duplicates:
        index = test_keys.index(key)
        block = test_blocks[index]
    end_time = time.time()
    return 1000*(end_time - start_time)
    
def get_model_size_from_details(kll):
    uni_bytes_string = kll.to_string().splitlines()[-4:-3][0]
    bytes_string = uni_bytes_string.encode('ascii','ignore')
    num_of_bytes = bytes_string[19:].strip()
    return num_of_bytes

def calculate_cost(histogram, total):
    cost = 0.0
    total = total * 1.0
    for miss in histogram.keys():
        cost += (miss+1)*(histogram[miss]/total)
    return cost - 1

def test(mapping):
    keys_without_duplicates = list(dict.fromkeys(test_keys))
    histogram = {}
    max_time = -1
    for key in keys_without_duplicates:
        est = search(key, mapping)
        index = get_index(test_keys, key)
        actual = test_blocks[index]
        # print("{},{}".format(est,actual))
        miss = abs(est - actual)
        if miss not in histogram:
            histogram[miss] = 1
        else:
            histogram[miss] = histogram[miss] + 1
        # print("For key: {}, est is {} and actual is {}".format(key,est,actual))
    return (histogram, len(keys_without_duplicates))

def search(key, mapping):
    quantile = get_quantile_of_element(mapping, key)
    estimated_block = choose_block_for_quantile(quantile)
    return estimated_block

def train(k):
    (quantiles, kll) = generate_quantiles(k)
    mapping = generate_qunatile_to_block_mapping(quantiles)
    return (mapping, kll)

def generate_quantiles(k):
    kll = kll_ints_sketch(k)
    kll.update(train_keys)
    quantiles = kll.get_quantiles(np.arange(0.0, 1.0, 1.0/number_of_blocks_in_memory))
    quantiles_without_duplicates = list(OrderedDict.fromkeys(quantiles))
    return (quantiles_without_duplicates, kll)

def generate_qunatile_to_block_mapping(quantiles):
    mapping = []

    for quantile in quantiles:
        d = (quantile, choose_block_for_quantile(quantile))
        mapping.append(d)
    return mapping

def choose_block_for_quantile(quantile):
    index = get_index(train_keys, quantile)
    block = train_blocks[index]
    return block

def get_quantile_of_element(quantiles, n):
    largest = [-1]
    for quantile in quantiles:
        if n < quantile[0]: # array is already sorted so we can stop after passing n
            return largest[0]
        largest = quantile
    return largest[0]

def get_index(a, x, lo=0, hi=None):
    # Just a simple binary search
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi)                  
    return pos if pos != hi and a[pos] == x else -1  

main()

# k --- Model size
# 8 - 380
# 9 - 364
# 10 - 388
# 11 - 396
# 12 - 388
# 44 - 608
# 60 - 760
# 1024 - 9324 = around (1kb) (try k=1024), try LI of size 1kb as well

