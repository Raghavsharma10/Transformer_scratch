def start_tag(self, indent):
		"""
		Generate the string for the element's start tag.
		"""
		return u"%s<%s%s/>" % (indent, self.tagName, u"".join(u" %s=\"%s\"" % keyvalue for keyvalue in self.attributes.items()))