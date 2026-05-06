def get_feedback(self, block = True, timeout = None):
		"""
		Gets the next feedback message.

		Each feedback message is a 2-tuple of (timestamp, device_token)."""
		if self._feedback_greenlet is None:
			self._feedback_greenlet = gevent.spawn(self._feedback_loop)
		return self._feedback_queue.get(block = block, timeout = timeout)