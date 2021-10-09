import glob
import statistics
from matplotlib import pyplot as plt

res = []

def get_data(name):
	with open(f'{name}.log', 'r') as f:
		sizes = []
		perfs = []
		for line in f:
			if 'Gflop/s' in line:
				size = int(line.split(' ')[1].split('\t')[0])
				perf = float(line.split(' ')[2])
				sizes.append(size)
				perfs.append(perf)

	return sizes, perfs

blas_sizes, blas_perfs = get_data('benchmark-blas')

for file in glob.glob('*.log'):
	name = file[:-4]
	sizes, perfs = get_data(name)
		
	plt.clf()
	plt.title(name)
	plt.xlabel('Matrix Size')
	plt.ylabel('Performance (GFlops)')
	plt.plot(blas_sizes, blas_perfs, '-bo', label='blas')
	plt.plot(sizes, perfs, '-rx', label=name)
	plt.legend()
	plt.savefig(f'{name}.png')