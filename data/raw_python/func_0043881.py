def community_colors(n):
	"""
	Returns a list of visually separable colors according to total communities
	"""

	if (n > 0):
		colors = cl.scales['12']['qual']['Paired']
		shuffle(colors)

		return colors[:n]
	else:
		return choice(cl.scales['12']['qual']['Paired'])