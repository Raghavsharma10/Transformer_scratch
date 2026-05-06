def get_tags(self, rev=None):
		"""
		Get the tags for the given revision specifier (or the
		current revision if not specified).
		"""
		rev_num = self._get_rev_num(rev)
		# rev_num might end with '+', indicating local modifications.
		return (
			set(self._read_tags_for_rev(rev_num))
			if not rev_num.endswith('+')
			else set([])
		)