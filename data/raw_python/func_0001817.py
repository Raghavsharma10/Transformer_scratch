def wait_send(self, timeout = None):
		"""Wait until all queued messages are sent."""
		self._send_queue_cleared.clear()
		self._send_queue_cleared.wait(timeout = timeout)