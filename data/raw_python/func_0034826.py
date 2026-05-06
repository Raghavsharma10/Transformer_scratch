def similar(self, threshold, **criterias):
		'''Find text-based field matches with similarity (1-levenshtein/length)
			higher than specified threshold (0 to 1, 1 being an exact match)'''
		# XXX: use F from https://docs.djangoproject.com/en/1.8/ref/models/expressions/
		meta = self.model._meta
		funcs, params = list(), list()
		for name,val in criterias.iteritems():
			name = meta.get_field(name, many_to_many=False).column
			name = '.'.join(it.imap(connection.ops.quote_name, (meta.db_table, name)))
			# Alas, pg_trgm is for containment tests, not fuzzy matches,
			#  but it can potentially be used to find closest results as well
			# funcs.append( 'similarity(CAST({0}.{1} as text), CAST(%s as text))'\
			# Ok, these two are just to make sure levenshtein() won't crash
			#  w/ "argument exceeds the maximum length of N bytes error"
			funcs.append('octet_length({0}) <= {1}'.format(name, self.levenshtein_limit))
			funcs.append('octet_length(%s) <= {0}'.format(self.levenshtein_limit))
			# Then there's a possibility of division by zero...
			funcs.append('length({0}) > 0'.format(name))
			# And if everything else fits, the comparison itself
			funcs.append('levenshtein({0}, %s) / CAST(length({0}) AS numeric) < %s'.format(name))
			params.extend((val, val, float(1 - threshold)))
		return self.extra(where=funcs, params=params)