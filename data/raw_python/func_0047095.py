def decodeTagAttributes(self, text):
		"""docstring for decodeTagAttributes"""
		attribs = {}
		if text.strip() == u'':
			return attribs
		scanner = _attributePat.scanner(text)
		match = scanner.search()
		while match:
			key, val1, val2, val3, val4 = match.groups()
			value = val1 or val2 or val3 or val4
			if value:
				value = _space.sub(u' ', value).strip()
			else:
				value = ''
			attribs[key] = self.decodeCharReferences(value)

			match = scanner.search()
		return attribs