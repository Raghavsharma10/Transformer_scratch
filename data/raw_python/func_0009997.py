def synchronous_command(self, *args, **kwargs):
		'''
		Forward a command to the remote chrome instance via the transport
		connection, and check the return for an error.

		If the command resulted in an error, a `ChromeController.ChromeError` is raised,
		with the error string containing the response from the remote
		chrome instance describing the problem and it's cause.

		Otherwise, the decoded json data-structure returned from the remote instance is
		returned.

		'''

		self.transport.check_process_ded()
		ret = self.transport.synchronous_command(tab_key=self.tab_id, *args, **kwargs)
		self.transport.check_process_ded()
		self.__check_ret(ret)
		self.transport.check_process_ded()
		return ret