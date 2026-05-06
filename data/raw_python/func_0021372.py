def auth_add_creds(self, username, password, pwtype='plain'):
		"""
		Add a valid set of credentials to be accepted for authentication.
		Calling this function will automatically enable requiring
		authentication. Passwords can be provided in either plaintext or
		as a hash by specifying the hash type in the *pwtype* argument.

		:param str username: The username of the credentials to be added.
		:param password: The password data of the credentials to be added.
		:type password: bytes, str
		:param str pwtype: The type of the *password* data, (plain, md5, sha1, etc.).
		"""
		if not isinstance(password, (bytes, str)):
			raise TypeError("auth_add_creds() argument 2 must be bytes or str, not {0}".format(type(password).__name__))
		pwtype = pwtype.lower()
		if not pwtype in ('plain', 'md5', 'sha1', 'sha256', 'sha384', 'sha512'):
			raise ValueError('invalid password type, must be \'plain\', or supported by hashlib')
		if self.__config.get('basic_auth') is None:
			self.__config['basic_auth'] = {}
			self.logger.info('basic authentication has been enabled')
		if pwtype != 'plain':
			algorithms_available = getattr(hashlib, 'algorithms_available', ()) or getattr(hashlib, 'algorithms', ())
			if pwtype not in algorithms_available:
				raise ValueError('hashlib does not support the desired algorithm')
			# only md5 and sha1 hex for backwards compatibility
			if pwtype == 'md5' and len(password) == 32:
				password = binascii.unhexlify(password)
			elif pwtype == 'sha1' and len(password) == 40:
				password = binascii.unhexlify(password)
			if not isinstance(password, bytes):
				password = password.encode('UTF-8')
			if len(hashlib.new(pwtype, b'foobar').digest()) != len(password):
				raise ValueError('the length of the password hash does not match the type specified')
		self.__config['basic_auth'][username] = {'value': password, 'type': pwtype}