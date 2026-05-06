def send(self, command, tab_key, params=None):
		'''
		Send command `command` with optional parameters `params` to the
		remote chrome instance.

		The command `id` is automatically added to the outgoing message.

		return value is the command id, which can be used to match a command
		to it's associated response.
		'''
		self.__check_open_socket(tab_key)

		sent_id = self.msg_id

		command = {
				"id": self.msg_id,
				"method": command,
			}

		if params:
			command["params"] = params
		navcom = json.dumps(command)


		# self.log.debug("		Sending: '%s'", navcom)
		try:
			self.soclist[tab_key].send(navcom)
		except (socket.timeout, websocket.WebSocketTimeoutException):
			raise cr_exceptions.ChromeCommunicationsError("Failure sending command to chromium.")
		except websocket.WebSocketConnectionClosedException:
			raise cr_exceptions.ChromeCommunicationsError("Websocket appears to have been closed. Is the"
				" remote chromium instance dead?")


		self.msg_id += 1
		return sent_id