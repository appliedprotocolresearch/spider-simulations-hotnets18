import sys
import pickle

with open(sys.argv[1], 'rb') as input:
    [all_paths, _] = pickle.load(input)

f = open(sys.argv[2], 'w')

for key in all_paths:
    src = key[0]
    dst = key[1]
    paths = all_paths[key]

    f.write("(" + str(src) + ", " + str(dst) + ")\n")
    for p in paths:
        f.write(str(p) + "\n")
    f.write("\n")


