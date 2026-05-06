def pattern_to_regex(cls, *args, **kw):
		"""
		Warn about deprecation.
		"""
		cls._deprecated()
		return super(GitIgnorePattern, cls).pattern_to_regex(*args, **kw)