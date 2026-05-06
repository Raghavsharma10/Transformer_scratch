def recv_all_filtered(self, keycheck, tab_key, timeout=0.5):
		'''
		Receive a all messages matching a filter, using the callable `keycheck` to filter received messages
		for content.

		This function will *ALWAY* block for at least `timeout` seconds.

		If chromium is for some reason continuously streaming responses, it may block forever!

		`keycheck` is expected to be a callable that takes a single parameter (the decoded response
		from chromium), and returns a boolean (true, if the command is the one filtered for, or false
		if the command is not the one filtered for).

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
		ret           = [tmp for tmp in self.messages[tab_key] if keycheck(tmp)]
		self.messages[tab_key] = [tmp for tmp in self.messages[tab_key] if not keycheck(tmp)]

		self.log.debug("Waiting for all messages from the socket")
		timeout_at = time.time() + timeout
		while 1:
			tmp = self.___recv(tab_key, timeout=timeout)
			if keycheck(tmp):
				ret.append(tmp)
			else:
				self.messages[tab_key].append(tmp)

			if time.time() > timeout_at:
				return ret
			else:
				self.log.debug("Sleeping: %s, %s" % (timeout_at, time.time()))
				time.sleep(0.005)