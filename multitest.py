import json
import torch
import argparse
from time import sleep
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

def sqr(x, y):
    sleep(0.01)
    print ("Sleep!")
    return x, x * y

def runner():
    l = list(range(100))
    ll = []
    results = [0] * 32
    cnt = 0
    with ThreadPoolExecutor(max_workers=5) as t:
        for i in range(30):
            idx = i + 2
            s = t.submit(sqr, l[idx], l[i])
            ll.append(s)

        for future in as_completed(ll):
            print ("Finish!", cnt)
            cnt += 1
            if cnt % 5 == 0:
                print ("Yes")
            a, b = future.result()
            results[a] = b

    return results

my_list = runner()
print (my_list)