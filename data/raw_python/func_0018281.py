def _send(self, email_message):
		"""A helper method that does the actual sending."""
		if not email_message.recipients():
			return False
		from_email = email_message.from_email
		recipients = email_message.recipients()
		try:
			self.connection.messages.create(
				to=recipients,
				from_=from_email,
				body=email_message.body
			)
		except Exception:
			if not self.fail_silently:
				raise
			return False
		return True