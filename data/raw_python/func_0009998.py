def drain_transport(self):
		'''
		"Drain" the transport connection.

		This command simply returns all waiting messages sent from the remote chrome
		instance. This can be useful when waiting for a specific asynchronous message
		from chrome, but higher level calls are better suited for managing wait-for-message
		type needs.

		'''
		self.transport.check_process_ded()
		ret = self.transport.drain(tab_key=self.tab_id)
		self.transport.check_process_ded()
		return ret