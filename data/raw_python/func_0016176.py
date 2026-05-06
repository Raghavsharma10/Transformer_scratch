def search_function(root1, q, s, f, l, o='g'):
	"""
	function to get links
	"""
	global links
	links = search(q, o, s, f, l)
	root1.destroy()
	root1.quit()