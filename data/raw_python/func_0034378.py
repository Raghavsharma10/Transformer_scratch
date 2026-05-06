def getsteps(levels, tagmax):
	""" Returns a list with the max number of posts per "tagcloud level"
	"""
	ntw = levels
	if ntw < 2:
		ntw = 2

	steps = [(stp, 1 + (stp * int(math.ceil(tagmax * 1.0 / ntw - 1))))
				for stp in range(ntw)]
	# just to be sure~
	steps[-1] = (steps[-1][0], tagmax+1)
	return steps