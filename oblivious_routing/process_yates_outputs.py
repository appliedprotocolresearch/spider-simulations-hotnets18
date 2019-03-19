""" read yates paths file, convert it to python object and pickle it for 
easy access later """

import cPickle as pickle 

def read_paths_file(filename):
	paths = {}

	with open(filename, 'r') as f:
		lines = [line.rstrip('\n') for line in f]

		for line in lines:
			if line:
				if line[0] == 'h':
					src = line.split(' ')[0]
					dest = line.split(' ')[2]
					src = int(src[1:]) + 50
					dest = int(dest[1:]) + 50
				else:
					path = line.split(' ')[1:-3]
					path_list = [src - 50]
					for edge in path:
						u = edge.split(',')[0]
						v = edge.split(',')[1]
						u = int(u[2:]) + 50
						v = int(v[1:-1]) + 50
						path_list.append(u)
					path_list.append(v)
					path_list.append(dest - 50)
					if (src - 50, dest - 50) not in paths.keys():
						paths[src - 50, dest - 50] = []
					paths[src - 50, dest - 50].append(path_list)
			else:
				pass

	print paths

	with open(filename + '.pkl', 'wb') as output:
		pickle.dump(paths, output, pickle.HIGHEST_PROTOCOL)

def main():
	read_paths_file('./sw_50_random_capacity')

if __name__=='__main__':
	main()