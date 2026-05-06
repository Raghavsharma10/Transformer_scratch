def recv(self, tab_key, message_id=None, timeout=30):
		'''
		Recieve a message, optionally filtering for a specified message id.

		If `message_id` is none, the first command in the receive queue is returned.
		If `message_id` is not none, the command waits untill a message is received with
		the specified id, or it times out.

		Timeout is the number of seconds to wait for a response, or `None` if the timeout
		has expired with no response.

		'''

		self.__check_open_socket(tab_key)

		# First, check if the message has already been received.
		for idx in range(len(self.messages[tab_key])):
			if self.messages[tab_key][idx]:
				if "id" in self.messages[tab_key][idx] and message_id:
					if self.messages[tab_key][idx]['id'] == message_id:
						return self.messages[tab_key].pop(idx)

		# Then spin untill we either have the message,
		# or have timed out.
		def check_func(message):
			if message_id is None:
				return True
			if not message:
				self.log.debug("Message is not true (%s)!", message)
				return False
			if "id" in message:
				return message['id'] == message_id
			return False

		return self.recv_filtered(check_func, tab_key, timeout)