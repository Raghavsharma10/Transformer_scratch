def auth_set(self, status):
		"""
		Enable or disable requiring authentication on all incoming requests.

		:param bool status: Whether to enable or disable requiring authentication.
		"""
		if not bool(status):
			self.__config['basic_auth'] = None
			self.logger.info('basic authentication has been disabled')
		else:
			self.__config['basic_auth'] = {}
			self.logger.info('basic authentication has been enabled')