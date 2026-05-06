def synchronous_command(self, command, tab_key, **params):
		"""
		Synchronously execute command `command` with params `params` in the
		remote chrome instance, returning the response from the chrome instance.

		"""
		self.log.debug("Synchronous_command to tab %s (%s):", tab_key, self._get_cr_tab_meta_for_key(tab_key))
		self.log.debug("	command: '%s'", command)
		self.log.debug("	params:  '%s'", params)
		self.log.debug("	tab_key:  '%s'", tab_key)

		send_id = self.send(command=command, tab_key=tab_key, params=params)
		resp = self.recv(message_id=send_id, tab_key=tab_key)

		self.log.debug("	Response: '%s'", str(resp).encode("ascii", 'ignore').decode("ascii"))

		# self.log.debug("	resolved tab idx %s:", self.tab_id_map[tab_key])
		return resp