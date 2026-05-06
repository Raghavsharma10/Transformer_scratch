def resolve_ssl_protocol_version(version=None):
	"""
	Look up an SSL protocol version by name. If *version* is not specified, then
	the strongest protocol available will be returned.

	:param str version: The name of the version to look up.
	:return: A protocol constant from the :py:mod:`ssl` module.
	:rtype: int
	"""
	if version is None:
		protocol_preference = ('TLSv1_2', 'TLSv1_1', 'TLSv1', 'SSLv3', 'SSLv23', 'SSLv2')
		for protocol in protocol_preference:
			if hasattr(ssl, 'PROTOCOL_' + protocol):
				return getattr(ssl, 'PROTOCOL_' + protocol)
		raise RuntimeError('could not find a suitable ssl PROTOCOL_ version constant')
	elif isinstance(version, str):
		if not hasattr(ssl, 'PROTOCOL_' + version):
			raise ValueError('invalid ssl protocol version: ' + version)
		return getattr(ssl, 'PROTOCOL_' + version)
	raise TypeError("ssl_version() argument 1 must be str, not {0}".format(type(version).__name__))