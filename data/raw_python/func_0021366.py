def remove_sni_cert(self, hostname):
		"""
		Remove the SSL Server Name Indicator (SNI) certificate configuration for
		the specified *hostname*.

		.. warning::

			This method will raise a :py:exc:`RuntimeError` if either the SNI
			extension is not available in the :py:mod:`ssl` module or if SSL was
			not enabled at initialization time through the use of arguments to
			:py:meth:`~.__init__`.

		.. versionadded:: 2.2.0

		:param str hostname: The hostname to delete the SNI configuration for.
		"""
		if not g_ssl_has_server_sni:
			raise RuntimeError('the ssl server name indicator extension is unavailable')
		if self._ssl_sni_entries is None:
			raise RuntimeError('ssl was not enabled on initialization')
		sni_entry = self._ssl_sni_entries.pop(hostname, None)
		if sni_entry is None:
			raise ValueError('the specified hostname does not have an sni certificate configuration')