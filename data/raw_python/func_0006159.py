def gender(self):
		''' Tries to scrape the correct gender for a given word from wordreference.com '''

		elements = self.tree.xpath('//table[@class="WRD"]')
		if len(elements):
			elements = self.tree.xpath('//table[@class="WRD"]')[0]
			if len(elements):
				if '/iten/' in self.page.url:
					elements = elements.xpath('//td[@class="FrWrd"]/em[@class="POS2"]/text()')
				elif '/enit/' in self.page.url:
					elements = elements.xpath('//td[@class="ToWrd"]/em[@class="POS2"]/text()')
				else:
					return [None]
				element = [element[1:] for element in elements if element in ['nm', 'nf']]
				counter = Counter(element)
				if len(counter.most_common(1)):
					result = counter.most_common(1)[0][0]
					return [result]
		return [None]