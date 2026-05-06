def format(self, sql, params):
		"""
		Formats the SQL query to use ordinal parameters instead of named
		parameters.

		*sql* (|string|) is the SQL query.

		*params* (|dict|) maps each named parameter (|string|) to value
		(|object|). If |self.named| is "numeric", then *params* can be
		simply a |sequence| of values mapped by index.

		Returns a 2-|tuple| containing: the formatted SQL query (|string|),
		and the ordinal parameters (|list|).
		"""
		if isinstance(sql, unicode):
			string_type = unicode
		elif isinstance(sql, bytes):
			string_type = bytes
			sql = sql.decode(_BYTES_ENCODING)
		else:
			raise TypeError("sql:{!r} is not a unicode or byte string.".format(sql))

		if self.named == 'numeric':
			if isinstance(params, collections.Mapping):
				params = {string_type(idx): val for idx, val in iteritems(params)}
			elif isinstance(params, collections.Sequence) and not isinstance(params, (unicode, bytes)):
				params = {string_type(idx): val for idx, val in enumerate(params, 1)}

		if not isinstance(params, collections.Mapping):
			raise TypeError("params:{!r} is not a dict.".format(params))

		# Find named parameters.
		names = self.match.findall(sql)

		# Map named parameters to ordinals.
		ord_params = []
		name_to_ords = {}
		for name in names:
			value = params[name]
			if isinstance(value, tuple):
				ord_params.extend(value)
				if name not in name_to_ords:
					name_to_ords[name] = '(' + ','.join((self.replace,) * len(value)) + ')'
			else:
				ord_params.append(value)
				if name not in name_to_ords:
					name_to_ords[name] = self.replace

		# Replace named parameters with ordinals.
		sql = self.match.sub(lambda m: name_to_ords[m.group(1)], sql)

		# Make sure the query is returned as the proper string type.
		if string_type is bytes:
			sql = sql.encode(_BYTES_ENCODING)

		# Return formatted SQL and new ordinal parameters.
		return sql, ord_params