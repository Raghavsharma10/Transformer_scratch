def pos(self):
		''' Tries to decide about the part of speech. '''

		tags = []
		if self.tree.xpath('//div[@class="grad733100"]/h2[@class="inline"]//text()'):
			info = self.tree.xpath('//div[@class="grad733100"]/h2[@class="inline"]')[0].text_content()
			info = info.strip('I ')
			if info.startswith(('de', 'het')):
				tags.append('NN')
			if not info.startswith(('de', 'het')) and info.endswith('en'):
				tags.append('VB')
			if not info.startswith(('de', 'het')) and not info.endswith('en'):
				tags.append('JJ')
		return tags