def start(self):
		"""Start the message sending loop."""
		if self._send_greenlet is None:
			self._send_greenlet = gevent.spawn(self._send_loop)