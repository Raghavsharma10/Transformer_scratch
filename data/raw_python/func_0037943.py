def get_parent_tags(self, rev=None):
		"""
		Return the tags for the parent revision (or None if no single
			parent can be identified).
		"""
		try:
			parent_rev = one(self.get_parent_revs(rev))
		except Exception:
			return None
		return self.get_tags(parent_rev)