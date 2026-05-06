def recv_filtered(self, keycheck, tab_key, timeout=30, message=None):
		'''
		Receive a filtered message, using the callable `keycheck` to filter received messages
		for content.

		`keycheck` is expected to be a callable that takes a single parameter (the decoded response
		from chromium), and returns a boolean (true, if the command is the one filtered for, or false
		if the command is not the one filtered for).

		This is used internally, for example, by `recv()`, to filter the response for a specific ID:

		```
			def check_func(message):
				if message_id is None:
					return True
				if "id" in message:
					return message['id'] == message_id
				return False
			return self.recv_filtered(check_func, timeout)

		```

		Note that the function is defined dynamically, and `message_id` is captured via closure.

		'''


		self.__check_open_socket(tab_key)

		# First, check if the message has already been received.
		for idx in range(len(self.messages[tab_key])):
			if keycheck(self.messages[tab_key][idx]):
				return self.messages[tab_key].pop(idx)

		timeout_at = time.time() + timeout
		while 1:
			tmp = self.___recv(tab_key)
			if keycheck(tmp):
				return tmp
			else:
				self.messages[tab_key].append(tmp)

			if time.time() > timeout_at:
				if message:
					raise cr_exceptions.ChromeResponseNotReceived("Failed to receive response in recv_filtered() (%s)" % message)
				else:
					raise cr_exceptions.ChromeResponseNotReceived("Failed to receive response in recv_filtered()")
			else:
				time.sleep(0.005)