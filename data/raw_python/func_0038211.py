def vary_radius(dt):
	"""Vary the disc radius over time"""
	global time
	time += dt
	disc.inner_radius = disc.outer_radius = 2.5 + math.sin(time / 2.0) * 1.5