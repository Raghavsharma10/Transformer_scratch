def show_filetypes(extensions):
	"""
	function to show valid file extensions
	"""
	for item in extensions.items():
		val = item[1]
		if type(item[1]) == list:
			val = ", ".join(str(x) for x in item[1])
		print("{0:4}: {1}".format(val, item[0]))