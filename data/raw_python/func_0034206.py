def get_id_parts(self):
		"""
		Return the three pieces of the int_8s-style sngl_inspiral
		event_id.
		"""
		int_event_id = int(self.event_id)
		a = int_event_id // 1000000000
		slidenum = (int_event_id % 1000000000) // 100000
		b = int_event_id % 100000
		return int(a), int(slidenum), int(b)