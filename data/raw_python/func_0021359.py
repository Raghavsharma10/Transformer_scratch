def close(self):
		"""
		Close the web socket connection and stop processing results. If the
		connection is still open, a WebSocket close message will be sent to the
		peer.
		"""
		if not self.connected:
			return
		self.connected = False
		if self.handler.wfile.closed:
			return
		if select.select([], [self.handler.wfile], [], 0)[1]:
			with self.lock:
				self.handler.wfile.write(b'\x88\x00')
		self.handler.wfile.flush()
		self.on_closed()