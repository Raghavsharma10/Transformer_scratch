def sort(self, *args):
		"""
		Sort the internal segment lists.  The optional args are
		passed to the .sort() method of the segment lists.  This
		can be used to control the sort order by providing an
		alternate comparison function.  The default is to sort by
		start time with ties broken by end time.
		"""
		self.valid.sort(*args)
		self.active.sort(*args)