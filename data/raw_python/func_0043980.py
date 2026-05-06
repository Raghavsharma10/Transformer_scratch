def get_range(self):
		""" Get range """

		if not self.page:
			return (1, self.last_blocks[self.coinid])

		# Get start of the range
		start = self.page * self.limit

		# Get finish of the range
		end = (self.page + 1) * self.limit

		if start > self.last_blocks[self.coinid]:
			return (1,1)
		if end > self.last_blocks[self.coinid]:
			return (start, self.last_blocks[self.coinid])
		return (start, end)