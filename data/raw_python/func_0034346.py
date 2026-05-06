def __setParentSymbol(self, value):
		"""self.__parentSymbol variable setter"""

		errors = []
		if not value is str and not value.split():
			errors.append('parentSymbol_ERROR : Symbol : must be char or string!')
		else:
			self.__parentSymbol = value

		if errors:
			view.Tli.showErrors('SymbolError', errors)