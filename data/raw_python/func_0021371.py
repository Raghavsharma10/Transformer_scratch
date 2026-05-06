def auth_delete_creds(self, username=None):
		"""
		Delete the credentials for a specific username if specified or all
		stored credentials.

		:param str username: The username of the credentials to delete.
		"""
		if not username:
			self.__config['basic_auth'] = {}
			self.logger.info('basic authentication database has been cleared of all entries')
			return
		del self.__config['basic_auth'][username]