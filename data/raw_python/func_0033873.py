def string_format_func(s):
	"""
	Function used internally to format string data for output to XML.
	Escapes back-slashes and quotes, and wraps the resulting string in
	quotes.
	"""
	return u"\"%s\"" % unicode(s).replace(u"\\", u"\\\\").replace(u"\"", u"\\\"")