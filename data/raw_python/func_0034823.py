def handler(self):
		'Parametrized handler function'
		return ft.partial(self.base.handler, parameter=self.parameter)\
			if self.parameter else self.base.handler