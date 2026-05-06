def parse(text, showToc=True):
	"""Returns HTML from MediaWiki markup"""
	p = Parser(show_toc=showToc)
	return p.parse(text)