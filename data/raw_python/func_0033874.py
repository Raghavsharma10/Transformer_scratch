def mk_complex_format_func(fmt):
	"""
	Function used internally to generate functions to format complex
	valued data.
	"""
	fmt = fmt + u"+i" + fmt
	def complex_format_func(z):
		return fmt % (z.real, z.imag)
	return complex_format_func