def pos(self, element = None):
		''' Tries to decide about the part of speech. '''

		tags = []
		if element:
			if re.search('[\w|\s]+ [m|f]\.', element, re.U):
				tags.append('NN')
			if '[VERB]' in element:
				tags.append('VB')
			if 'adj.' in element and re.search('([\w|\s]+, [\w|\s]+)', element, re.U):
				tags.append('JJ')
		else:
			for element in self.elements:
				if element.startswith(self.word):
					tags += self.pos(element)
		return list(set(tags))