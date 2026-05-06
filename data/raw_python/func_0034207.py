def get_slide_number(self):
		"""
		Return the slide-number for this trigger
		"""
		a, slide_number, b = self.get_id_parts()
		if slide_number > 5000:
			slide_number = 5000 - slide_number
		return slide_number