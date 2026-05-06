def parse(self, tokens):
		'''Parses a list of tokens into a JSON-serializable object.

		The parsing proceeds from left to right and is greedy.

		Precedence order:
		  1. Parameters with active context. For example, an Option with
		     nargs=-1 will gobble all the remaining tokens.
		  2. Subcommands.
		  3. Parameters.

		The keys of the returned object are the names of parameters or
		subcommands. Subcommands are encoded as nested objects. Multiple
		parameters are encoded as lists. All other values are encoded as
		parameter-specified data types, or strings if not specified.
		'''
		self._ping_main()

		# Pre-parsing:
		#   1. Expand globbed options: -abc --> -a -b -c
		def is_globbed(s):
			return len(s) > 2 and s.startswith('-') and not s.startswith('--')
		expanded = [["-" + c for c in list(token[1:])] if is_globbed(token) else [token] for token in tokens]

		# Parsing: pass off to main command after flattening expanded tokens list
		return self._main.parse(list(itertools.chain.from_iterable(expanded)))