def getslide(self,slide_num):
		"""
		Return the triggers with a specific slide number.
		@param slide_num: the slide number to recover (contained in the event_id)
		"""
		slideTrigs = self.copy()
		slideTrigs.extend(row for row in self if row.get_slide_number() == slide_num)
		return slideTrigs