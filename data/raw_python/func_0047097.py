def safeEncodeAttribute(self, encValue):
		"""docstring for safeEncodeAttribute"""
		encValue = encValue.replace(u'&', u'&amp;')
		encValue = encValue.replace(u'<', u'&lt;')
		encValue = encValue.replace(u'>', u'&gt;')
		encValue = encValue.replace(u'"', u'&quot;')
		encValue = encValue.replace(u'{', u'&#123;')
		encValue = encValue.replace(u'[', u'&#91;')
		encValue = encValue.replace(u"''", u'&#39;&#39;')
		encValue = encValue.replace(u'ISBN', u'&#73;SBN')
		encValue = encValue.replace(u'RFC', u'&#82;FC')
		encValue = encValue.replace(u'PMID', u'&#80;MID')
		encValue = encValue.replace(u'|', u'&#124;')
		encValue = encValue.replace(u'__', u'&#95;_')
		encValue = encValue.replace(u'\n', u'&#10;')
		encValue = encValue.replace(u'\r', u'&#13;')
		encValue = encValue.replace(u'\t', u'&#9;')
		return encValue