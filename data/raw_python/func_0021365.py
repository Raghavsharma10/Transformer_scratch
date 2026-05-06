def add_sni_cert(self, hostname, ssl_certfile=None, ssl_keyfile=None, ssl_version=None):
		"""
		Add an SSL certificate for a specific hostname as supported by SSL's
		Server Name Indicator (SNI) extension. See :rfc:`3546` for more details
		on SSL extensions. In order to use this method, the server instance must
		have been initialized with at least one address configured for SSL.

		.. warning::

			This method will raise a :py:exc:`RuntimeError` if either the SNI
			extension is not available in the :py:mod:`ssl` module or if SSL was
			not enabled at initialization time through the use of arguments to
			:py:meth:`~.__init__`.

		.. versionadded:: 2.0.0

		:param str hostname: The hostname for this configuration.
		:param str ssl_certfile: An SSL certificate file to use, setting this enables SSL.
		:param str ssl_keyfile: An SSL certificate file to use.
		:param ssl_version: The SSL protocol version to use.
		"""
		if not g_ssl_has_server_sni:
			raise RuntimeError('the ssl server name indicator extension is unavailable')
		if self._ssl_sni_entries is None:
			raise RuntimeError('ssl was not enabled on initialization')
		if ssl_certfile:
			ssl_certfile = os.path.abspath(ssl_certfile)
		if ssl_keyfile:
			ssl_keyfile = os.path.abspath(ssl_keyfile)
		cert_info = SSLSNICertificate(hostname, ssl_certfile, ssl_keyfile)
		if ssl_version is None or isinstance(ssl_version, str):
			ssl_version = resolve_ssl_protocol_version(ssl_version)
		ssl_ctx = ssl.SSLContext(ssl_version)
		ssl_ctx.load_cert_chain(ssl_certfile, keyfile=ssl_keyfile)
		self._ssl_sni_entries[hostname] = SSLSNIEntry(context=ssl_ctx, certificate=cert_info)