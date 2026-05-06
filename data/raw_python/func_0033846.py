def make_parser(handler):
	"""
	Convenience function to construct a document parser with namespaces
	enabled and validation disabled.  Document validation is a nice
	feature, but enabling validation can require the LIGO LW DTD to be
	downloaded from the LDAS document server if the DTD is not included
	inline in the XML.  This requires a working connection to the
	internet and the server to be up.
	"""
	parser = sax.make_parser()
	parser.setContentHandler(handler)
	parser.setFeature(sax.handler.feature_namespaces, True)
	parser.setFeature(sax.handler.feature_validation, False)
	parser.setFeature(sax.handler.feature_external_ges, False)
	return parser