def send_messages(self, sms_messages):
		"""
		Receives a list of SMSMessage instances and returns a list of RQ `Job` instances.
		"""
		results = []
		for message in sms_messages:
			try:
				assert message.connection is None
			except AssertionError:
				if not self.fail_silently:
					raise
			backend = self.backend
			fail_silently = self.fail_silently
			result = django_rq.enqueue(self._send, message, backend=backend, fail_silently=fail_silently)
			results.append(result)
		return results