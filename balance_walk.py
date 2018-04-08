import sys
import numpy as np
import random

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
    capacity = float(sys.argv[2])
    p = float(sys.argv[3])
    q = 1 - p

    steps_hist = []
    for i in range(runs):
        steps = run_balance_walk(capacity, p, q)
        steps_hist.append(steps)
    c = capacity
    expected_steps = ((1 - (q/p)**(c/2.0))*c/(1 - (q/p)**c) - (c/2))/(p - q)
    observed_steps = np.average(steps_hist)

    print "expected steps:", expected_steps, "observed steps:", observed_steps

main()
