def formatmany(self, sql, many_params):
		"""
		Formats the SQL query to use ordinal parameters instead of named
		parameters.

		*sql* (|string|) is the SQL query.

		*many_params* (|iterable|) contains each *params* to format.

		- *params* (|dict|) maps each named parameter (|string|) to value
		  (|object|). If |self.named| is "numeric", then *params* can be
		  simply a |sequence| of values mapped by index.

		Returns a 2-|tuple| containing: the formatted SQL query (|string|),
		and a |list| containing each ordinal parameters (|list|).
		"""
		if isinstance(sql, unicode):
			string_type = unicode
		elif isinstance(sql, bytes):
			string_type = bytes
			sql = sql.decode(_BYTES_ENCODING)
		else:
			raise TypeError("sql:{!r} is not a unicode or byte string.".format(sql))

		if not isinstance(many_params, collections.Iterable) or isinstance(many_params, (unicode, bytes)):
			raise TypeError("many_params:{!r} is not iterable.".format(many_params))

		# Find named parameters.
		names = self.match.findall(sql)
		name_set = set(names)

		# Map named parameters to ordinals.
		many_ord_params = []
		name_to_ords = {}
		name_to_len = {}
		repl_str = self.replace
		repl_tuple = (repl_str,)
		for i, params in enumerate(many_params):
			if self.named == 'numeric':
				if isinstance(params, collections.Mapping):
					params = {string_type(idx): val for idx, val in iteritems(params)}
				elif isinstance(params, collections.Sequence) and not isinstance(params, (unicode, bytes)):
					params = {string_type(idx): val for idx, val in enumerate(params, 1)}

			if not isinstance(params, collections.Mapping):
				raise TypeError("many_params[{}]:{!r} is not a dict.".format(i, params))

			if not i: # first
				# Map names to ordinals, and determine what names are tuples and
				# what their lengths are.
				for name in name_set:
					value = params[name]
					if isinstance(value, tuple):
						tuple_len = len(value)
						name_to_ords[name] = '(' + ','.join(repl_tuple * tuple_len) + ')'
						name_to_len[name] = tuple_len
					else:
						name_to_ords[name] = repl_str
						name_to_len[name] = None

			# Make sure tuples match up and collapse tuples into ordinals.
			ord_params = []
			for name in names:
				value = params[name]
				tuple_len = name_to_len[name]
				if tuple_len is not None:
					if not isinstance(value, tuple):
						raise TypeError("many_params[{}][{!r}]:{!r} was expected to be a tuple.".format(i, name, value))
					elif len(value) != tuple_len:
						raise ValueError("many_params[{}][{!r}]:{!r} length was expected to be {}.".format(i, name, value, tuple_len))
					ord_params.extend(value)
				else:
					ord_params.append(value)
			many_ord_params.append(ord_params)

		# Replace named parameters with ordinals.
		sql = self.match.sub(lambda m: name_to_ords[m.group(1)], sql)

		# Make sure the query is returned as the proper string type.
		if string_type is bytes:
			sql = sql.encode(_BYTES_ENCODING)

		# Return formatted SQL and new ordinal parameters.
		return sql, many_ord_params