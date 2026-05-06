def strip_msa_100(msa, threshold, plot = False):
	"""
	strip out columns of a MSA that represent gaps for X percent (threshold) of sequences
	"""
	msa = [seq for seq in parse_fasta(msa)]
	columns = [[0, 0] for pos in msa[0][1]] # [[#bases, #gaps], [#bases, #gaps], ...]
	for seq in msa:
		for position, base in enumerate(seq[1]):
			if base == '-' or base == '.':
				columns[position][1] += 1
			else:
				columns[position][0] += 1
	columns = [float(float(g)/float(g+b)*100) for b, g in columns] # convert to percent gaps
	for seq in msa:
		stripped = []
		for position, base in enumerate(seq[1]):
			if columns[position] < threshold:
				stripped.append(base)
		yield [seq[0], ''.join(stripped)]
	if plot is not False:
		plot_gaps(plot, columns)