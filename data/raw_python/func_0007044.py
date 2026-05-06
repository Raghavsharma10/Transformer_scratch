def create_syslog(self,
		service_id,
		version_number,
		name,
		address,
		port=514,
		use_tls="0",
		tls_ca_cert=None,
		token=None,
		_format=None,
		response_condition=None):
		"""Create a Syslog for a particular service and version."""
		body = self._formdata({
			"name": name,
			"address": address,
			"port": port,
			"use_tls": use_tls,
			"tls_ca_cert": tls_ca_cert,
			"token": token,
			"format": _format,
			"response_condition": response_condition,
		}, FastlySyslog.FIELDS)
		content = self._fetch("/service/%s/version/%d/syslog" % (service_id, version_number), method="POST", body=body)
		return FastlySyslog(self, content)