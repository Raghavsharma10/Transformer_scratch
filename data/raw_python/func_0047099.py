def closeParagraph(self, mLastSection):
		"""Used by doBlockLevels()"""
		result = u''
		if mLastSection != u'':
			result = u'</' + mLastSection + u'>\n'

		return result