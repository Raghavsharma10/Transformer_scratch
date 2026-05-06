def message(self):
		"""
		*message* (:class:`str`) is the error message.
		"""
		return "Real path {real!r} was encountered at {first!r} and then {second!r}.".format(
			real=self.real_path,
			first=self.first_path,
			second=self.second_path,
		)