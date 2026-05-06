def ifocut(self, ifo, inplace=False):
		"""
		Return a SnglInspiralTable with rows from self having IFO equal
		to the given ifo. If inplace, modify self directly, else create
		a new table and fill it.
		"""
		if inplace:
			iterutils.inplace_filter(lambda row: row.ifo == ifo, self)
			return self
		else:
			ifoTrigs = self.copy()
			ifoTrigs.extend([row for row in self if row.ifo == ifo])
			return ifoTrigs