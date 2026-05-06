def update_syslog(self, service_id, version_number, name_key, **kwargs):
		"""Update the Syslog for a particular service and version."""
		body = self._formdata(kwargs, FastlySyslog.FIELDS)
		content = self._fetch("/service/%s/version/%d/syslog/%s" % (service_id, version_number, name_key), method="PUT", body=body)
		return FastlySyslog(self, content)