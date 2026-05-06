def consume(self, tokens):
		'''Have this parameter consume some tokens.

		This stores the consumed value for later use and returns the
		modified tokens array for further processing.
		'''
		n = len(tokens) if self._nargs == -1 else self._nargs
		if n > len(tokens):
			exit('Error: Not enough arguments for "{}".'.format(self._name), True)
		try:
			consumed = [self._type(e) if self._type is not None else e for e in tokens[:n]]
		except ValueError as e:
			exit('Error: Invalid type given to "{}", expected {}.'.format(
					self._name, self._type.__name__), True)
		if n == 1 and self._nargs == 1:
			consumed = consumed[0]
		self.post_consume(consumed)
		return tokens[n:]