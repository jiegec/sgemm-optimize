import glob

res = []

for file in glob.glob('*.log'):
	with open(file, 'r') as f:
		data = []
		for line in f:
			if 'Gflop/s' in line:
				perf = float(line.split(' ')[2])
				data.append(perf)
		avg = sum(data) / len(data)
		res.append((file, avg))

res = list(sorted(res, key=lambda k: k[1]))

for file, avg in res:
	print(f'{file}: {avg:.2f}')