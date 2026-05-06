def apply_to_segmentlist(self, seglist):
		"""
		Apply our low and high windows to the segments in a
		segmentlist.
		"""
		for i, seg in enumerate(seglist):
			seglist[i] = seg.__class__(seg[0] - self.low_window, seg[1] + self.high_window)