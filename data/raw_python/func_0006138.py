def reference(language, word):
	''' Returns the articles (singular and plural) combined with singular and plural for a given noun. '''

	sg, pl, art = word, '/'.join(plural(language, word)  or ['-']), [[''], ['']]
	art[0], art[1] = articles(language, word) or (['-'], ['-'])
	result = ['%s %s' % ('/'.join(art[0]), sg), '%s %s' % ('/'.join(art[1]), pl)]
	result = [None if x == '- -' else x for x in result]
	return result