def stop(self, timeout = 10.0):
		"""
		Send all pending messages, close connection.
		Returns True if no message left to sent. False if dirty.
		
		- timeout: seconds to wait for sending remaining messages. disconnect
		  immedately if None.
		"""
		if (self._send_greenlet is not None) and \
				(self._send_queue.qsize() > 0):
			self.wait_send(timeout = timeout)

		if self._send_greenlet is not None:
			gevent.kill(self._send_greenlet)
			self._send_greenlet = None
		if self._error_greenlet is not None:
			gevent.kill(self._error_greenlet)
			self._error_greenlet = None
		if self._feedback_greenlet is not None:
			gevent.kill(self._feedback_greenlet)
			self._feedback_greenlet = None

		return self._send_queue.qsize() < 1