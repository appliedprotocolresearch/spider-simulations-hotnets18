import sys
import numpy as np
import random
import itertools

def run_balance_walk(capacity, p, q):
    cur_balance = capacity/2.0
    steps = 0

    while (cur_balance != 0 and cur_balance != capacity):
        rand = random.random()
        if (rand < p):
            cur_balance += 1
        else:
            cur_balance -= 1
        steps += 1
    return steps

def main():
    runs = int(sys.argv[1])
    capacity_list = [10, 20, 40, 60, 80, 120, 140, 150, 170, 180, 200]
    p_list = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    for p, capacity in itertools.product(p_list, capacity_list): 
        q = 1 - p
        steps_hist = []
        for i in range(runs):
            steps = run_balance_walk(capacity, p, q)
            steps_hist.append(steps)
        c = capacity
        expected_steps = ((1 - (q/p)**(c/2.0))*c/(1 - (q/p)**c) - (c/2))/(p - q)
        observed_steps = np.average(steps_hist)
        error = abs(observed_steps - expected_steps)/expected_steps
        print "expected steps:", expected_steps, " observed steps:", observed_steps, " error:", error

main()
