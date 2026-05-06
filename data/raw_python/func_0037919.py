def set(self, **args) :
		"set multiple values quickly, ex : name = woopy"
		for k, v in args.items() :
			setattr(self, k, v)