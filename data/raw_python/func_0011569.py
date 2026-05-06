def match(self, files):
		"""
		Matches this pattern against the specified files.

		*files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
		each file relative to the root directory (e.g., ``"relative/path/to/file"``).

		Returns an :class:`~collections.abc.Iterable` yielding each matched
		file path (:class:`str`).
		"""
		raise NotImplementedError("{}.{} must override match().".format(self.__class__.__module__, self.__class__.__name__))