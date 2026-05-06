def __setWildcardSymbol(self, value):
		"""self.__wildcardSymbol variable setter"""

		errors = []
		if not value is str and not value.split():
			errors.append('wildcardSymbol_ERROR : Symbol : must be char or string!')
		else:
			self.__wildcardSymbol = value

		if errors:
			view.Tli.showErrors('SymbolError', errors)