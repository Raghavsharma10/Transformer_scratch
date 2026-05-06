def checkCss(self, value):
		"""docstring for checkCss"""
		stripped = self.decodeCharReferences(value)

		stripped = _cssCommentPat.sub(u'', stripped)
		value = stripped

		stripped = _toUTFPat.sub(self._convertToUtf8, stripped)
		stripped.replace(u'\\', u'')
		if _hackPat.search(stripped):
			# someone is haxx0ring
			return False

		return value