def pos(self, element = None):
		''' Tries to decide about the part of speech. '''

		tags = []
		if element:
			if element.startswith(('der', 'die', 'das')):
				tags.append('NN')
			if ' VERB' in element:
				tags.append('VB')
			if ' ADJ' in element:
				tags.append('JJ')
		else:
			for element in self.elements:
				if self.word in unicode(element):
					return self.pos(element)
		return tags