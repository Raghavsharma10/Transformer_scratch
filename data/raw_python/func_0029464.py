def plot_gaps(plot, columns):
	"""
	plot % of gaps at each position
	"""
	from plot_window import window_plot_convolve as plot_window
#	plot_window([columns], len(columns)*.01, plot)
	plot_window([[100 - i for i in columns]], len(columns)*.01, plot)