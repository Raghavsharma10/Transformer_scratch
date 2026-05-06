def validateTagAttributes(self, attribs, element):
		"""docstring for validateTagAttributes"""
		out = {}
		if element not in _whitelist:
			return out
		whitelist = _whitelist[element]
		for attribute in attribs:
			value = attribs[attribute]
			if attribute not in whitelist:
				continue
			# Strip javascript "expression" from stylesheets.
			# http://msdn.microsoft.com/workshop/author/dhtml/overview/recalc.asp
			if attribute == u'style':
				value = self.checkCss(value)
				if value == False:
					continue
			elif attribute == u'id':
				value = self.escapeId(value)
			# If this attribute was previously set, override it.
			# Output should only have one attribute of each name.
			out[attribute] = value
		return out