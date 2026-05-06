def sni_certs(self):
		"""
		.. versionadded:: 2.2.0

		:return: Return a tuple of :py:class:`~.SSLSNICertificate` instances for each of the certificates that are configured.
		:rtype: tuple
		"""
		if not g_ssl_has_server_sni or self._ssl_sni_entries is None:
			return tuple()
		return tuple(entry.certificate for entry in self._ssl_sni_entries.values())