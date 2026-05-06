def openList(self, char, mLastSection):
		"""
		These next three functions open, continue, and close the list
		element appropriate to the prefix character passed into them.
		"""
		result = self.closeParagraph(mLastSection)

		mDTopen = False
		if char == u'*':
			result += u'<ul><li>'
		elif char == u'#':
			result += u'<ol><li>'
		elif char == u':':
			result += u'<dl><dd>'
		elif char == u';':
			result += u'<dl><dt>'
			mDTopen = True
		else:
			result += u'<!-- ERR 1 -->'

		return result, mDTopen