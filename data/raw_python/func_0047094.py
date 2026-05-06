def removeHtmlComments(self, text):
		"""remove <!-- text --> comments from given text"""
		sb = []
		start = text.find(u'<!--')
		last = 0
		while start != -1:
			end = text.find(u'-->', start)
			if end == -1:
				break
			end += 3

			spaceStart = max(0, start-1)
			spaceEnd = end
			while text[spaceStart] == u' ' and spaceStart > 0:
				spaceStart -= 1
			while text[spaceEnd] == u' ':
				spaceEnd += 1

			if text[spaceStart] == u'\n' and text[spaceEnd] == u'\n':
				sb.append(text[last:spaceStart])
				sb.append(u'\n')
				last = spaceEnd+1
			else:
				sb.append(text[last:spaceStart+1])
				last = spaceEnd

			start = text.find(u'<!--', end)
		sb.append(text[last:])
		return u''.join(sb)