def _promote(self, name, instantiate=True):
		"""Create a new subclass of Context which incorporates instance attributes and new descriptors.
		
		This promotes an instance and its instance attributes up to being a class with class attributes, then
		returns an instance of that class.
		"""
		
		metaclass = type(self.__class__)
		contents = self.__dict__.copy()
		cls = metaclass(str(name), (self.__class__, ), contents)
		
		if instantiate:
			return cls()
		
		return cls