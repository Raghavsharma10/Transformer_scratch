def send_message(self, opcode, message):
		"""
		Send a message to the peer over the socket.

		:param int opcode: The opcode for the message to send.
		:param bytes message: The message data to send.
		"""
		if not isinstance(message, bytes):
			message = message.encode('utf-8')
		length = len(message)
		if not select.select([], [self.handler.wfile], [], 0)[1]:
			self.logger.error('the socket is not ready for writing')
			self.close()
			return
		buffer = b''
		buffer += struct.pack('B', 0x80 + opcode)
		if length <= 125:
			buffer += struct.pack('B', length)
		elif 126 <= length <= 65535:
			buffer += struct.pack('>BH', 126, length)
		else:
			buffer += struct.pack('>BQ', 127, length)
		buffer += message
		self._last_sent_opcode = opcode
		self.lock.acquire()
		try:
			self.handler.wfile.write(buffer)
			self.handler.wfile.flush()
		except Exception:
			self.logger.error('an error occurred while sending a message', exc_info=True)
			self.close()
		finally:
			self.lock.release()