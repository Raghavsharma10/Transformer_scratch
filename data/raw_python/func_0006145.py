def _extract(self, identifier):
		''' Extracts data from conjugation table. '''

		conjugation = []
		if self.tree.xpath('//p/b[normalize-space(text()) = "' + identifier.decode('utf-8') + '"]'):
			p = self.tree.xpath('//p/b[normalize-space(text()) = "' + identifier.decode('utf-8') + '"]')[0].getparent()
			for font in p.iterfind('font'):
				text = self._normalize(font.text_content())
				next = font.getnext()
				text += ' ' + self._normalize(next.text_content())
				while True:
					next = next.getnext()
					if next.tag != 'span':
						break
					text += '/' + self._normalize(next.text_content())
				conjugation.append(text)
		return conjugation