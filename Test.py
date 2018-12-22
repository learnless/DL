# -*- coding:utf-8 -*-
import random

if __name__=="__main__":
    num = [[random.uniform(0,1) for c in range(2)] for n in range(10)]
    print(num)