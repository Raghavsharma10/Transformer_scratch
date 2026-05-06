def open(self):
		"""
		Ensures we have a connection to the SMS gateway. Returns whether or not a new connection was required (True or False).
		"""
		if self.connection:
			# Nothing to do if the connection is already open.
			return False

		self.connection = self._get_twilio_client()
		return True