def articles(word):
	''' Returns the articles (singular and plural) for a given noun. '''

	from pattern.it import article

	result = [[None], [None]]
	genus = gender(word) or 'f'
	result[0] = [article(word, function = 'definite', gender = genus)]
	result[1] = [article(plural(word)[0], function = 'definite', gender = (genus, 'p'))]
	return result