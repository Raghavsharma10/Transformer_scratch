def optimize(self):
		"""
		Identifies segment lists that differ only in their
		instruments --- they have the same valid and active
		segments, the same name, version and the same comment ---
		and then deletes all but one of them, leaving just a single
		list having the union of the instruments.
		"""
		self.sort()
		segment_lists = dict(enumerate(self))
		for target, source in [(idx_a, idx_b) for (idx_a, seglist_a), (idx_b, seglist_b) in iterutils.choices(segment_lists.items(), 2) if seglist_a.valid == seglist_b.valid and seglist_a.active == seglist_b.active and seglist_a.name == seglist_b.name and seglist_a.version == seglist_b.version and seglist_a.comment == seglist_b.comment]:
			try:
				source = segment_lists.pop(source)
			except KeyError:
				continue
			segment_lists[target].instruments |= source.instruments
		self.clear()
		self.update(segment_lists.values())