def merge(self, elements):
		''' Merges all scraping results to a list sorted by frequency of occurrence. '''

		from collections import Counter
		from lltk.utils import list2tuple, tuple2list
		# The list2tuple conversion is necessary because mutable objects (e.g. lists) are not hashable
		merged = tuple2list([value for value, count in Counter(list2tuple(list(elements))).most_common()])
		return merged