def check_authorization(self):
		"""
		Check for the presence of a basic auth Authorization header and
		if the credentials contained within in are valid.

		:return: Whether or not the credentials are valid.
		:rtype: bool
		"""
		try:
			store = self.__config.get('basic_auth')
			if store is None:
				return True
			auth_info = self.headers.get('Authorization')
			if not auth_info:
				return False
			auth_info = auth_info.split()
			if len(auth_info) != 2 or auth_info[0] != 'Basic':
				return False
			auth_info = base64.b64decode(auth_info[1]).decode(sys.getdefaultencoding())
			username = auth_info.split(':')[0]
			password = ':'.join(auth_info.split(':')[1:])
			password_bytes = password.encode(sys.getdefaultencoding())
			if hasattr(self, 'custom_authentication'):
				if self.custom_authentication(username, password):
					self.basic_auth_user = username
					return True
				return False
			if not username in store:
				self.server.logger.warning('received invalid username: ' + username)
				return False
			password_data = store[username]

			if password_data['type'] == 'plain':
				if password == password_data['value']:
					self.basic_auth_user = username
					return True
			elif hashlib.new(password_data['type'], password_bytes).digest() == password_data['value']:
				self.basic_auth_user = username
				return True
			self.server.logger.warning('received invalid password from user: ' + username)
		except Exception:
			pass
		return False