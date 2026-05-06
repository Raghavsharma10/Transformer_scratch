def pos(self):
		''' Tries to decide about the part of speech. '''

		tags = []
		if self.tree.xpath('//div[@id="mw-content-text"]//a[@title="Hilfe:Wortart"]/text()'):
			info = self.tree.xpath('//div[@id="mw-content-text"]//a[@title="Hilfe:Wortart"]/text()')[0]
			if info == 'Substantiv':
				tags.append('NN')
			if info == 'Verb':
				tags.append('VB')
			if info == 'Adjektiv':
				tags.append('JJ')
		return tags