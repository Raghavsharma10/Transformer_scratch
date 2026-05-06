def get_serialize_format(self, mimetype):
		""" Get the serialization format for the given mimetype """
		format = self.formats.get(mimetype, None)
		if format is None:
			format = formats.get(mimetype, None)
		return format