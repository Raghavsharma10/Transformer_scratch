def add_format(mimetype, format, requires_context=False):
	""" Registers a new format to be used in a graph's serialize call
	    If you've installed an rdflib serializer plugin, use this
	    to add it to the content negotiation system
	    Set requires_context=True if this format requires a context-aware graph
	"""
	global formats
	global ctxless_mimetypes
	global all_mimetypes
	formats[mimetype] = format
	if not requires_context:
		ctxless_mimetypes.append(mimetype)
	all_mimetypes.append(mimetype)