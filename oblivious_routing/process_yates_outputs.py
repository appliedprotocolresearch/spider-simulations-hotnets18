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
					src = int(src[1:])
					dest = int(dest[1:])
				else:
					path = line.split(' ')[1:-3]
					path_list = []
					for edge in path:
						u = edge.split(',')[0]
						v = edge.split(',')[1]
						u = int(u[2:])
						v = int(v[1:-1])
						path_list.append(u)
					path_list.append(v)
					if (src, dest) not in paths.keys():
						paths[src, dest] = []
					paths[src, dest].append(path_list)
			else:
				pass

	with open(filename + '.pkl', 'wb') as output:
		pickle.dump(paths, output, pickle.HIGHEST_PROTOCOL)

def main():
	read_paths_file('./raeke_0')

if __name__=='__main__':
	main()