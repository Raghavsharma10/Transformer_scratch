def get_tags(self, rev=None):
		"""
		Return the tags for the current revision as a set
		"""
		rev = rev or 'HEAD'
		return set(self._invoke('tag', '--points-at', rev).splitlines())