import glob
import statistics

res = []

for file in glob.glob('*.log'):
	with open(file, 'r') as f:
		data = []
		for line in f:
			if 'Gflop/s' in line:
				perf = float(line.split(' ')[2])
				data.append(perf)
		avg = statistics.mean(data)
		stdev = statistics.stdev(data)
		res.append((file, avg, stdev, len(data)))

res = list(sorted(res, key=lambda k: k[1]))

for file, avg, stdev, count in res:
	print(f'{file}: {avg:.2f}(stdev={stdev:.2f}) of {count} matrix')