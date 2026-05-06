def pos(self, element = None):
		''' Tries to decide about the part of speech. '''

		tags = []
		if element:
			if element.startswith(('de ', 'het ', 'het/de', 'de/het')) and not re.search('\[[\w|\s][\w|\s]+\]', element.split('\r\n')[0], re.U):
				tags.append('NN')
			if re.search('[\w|\s|/]+ \| [\w|\s|/]+ - [\w|\s|/]+', element, re.U):
				tags.append('VB')
			if re.search('[\w|\s]+ \| [\w|\s]+', element, re.U):
				tags.append('JJ')
			return tags
		else:
			for element in self.elements:
				if self.word in unicode(element):
					tag = self.pos(element)
					if tag:
						return tag