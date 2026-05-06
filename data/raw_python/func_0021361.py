def on_message(self, opcode, message):
		"""
		The primary dispatch function to handle incoming WebSocket messages.

		:param int opcode: The opcode of the message that was received.
		:param bytes message: The data contained within the message.
		"""
		self.logger.debug("processing {0} (opcode: 0x{1:02x}) message".format(self._opcode_names.get(opcode, 'UNKNOWN'), opcode))
		if opcode == self._opcode_close:
			self.close()
		elif opcode == self._opcode_ping:
			if len(message) > 125:
				self.close()
				return
			self.send_message(self._opcode_pong, message)
		elif opcode == self._opcode_pong:
			pass
		elif opcode == self._opcode_binary:
			self.on_message_binary(message)
		elif opcode == self._opcode_text:
			try:
				message = self._decode_string(message)
			except UnicodeDecodeError:
				self.logger.warning('closing connection due to invalid unicode within a text message')
				self.close()
			else:
				self.on_message_text(message)
		elif opcode == self._opcode_continue:
			self.close()
		else:
			self.logger.warning("received unknown opcode: {0} (0x{0:02x})".format(opcode))
			self.close()