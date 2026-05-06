def create_request_setting(self,
		service_id,
		version_number,
		name,
		default_host=None,
		force_miss=None,
		force_ssl=None,
		action=None,
		bypass_busy_wait=None,
		max_stale_age=None,
		hash_keys=None,
		xff=None,
		timer_support=None,
		geo_headers=None,
		request_condition=None):
		"""Creates a new Request Settings object."""
		body = self._formdata({
			"name": name,
			"default_host": default_host,
			"force_miss": force_miss,
			"force_ssl": force_ssl,
			"action": action,
			"bypass_busy_wait": bypass_busy_wait,
			"max_stale_age": max_stale_age,
			"hash_keys": hash_keys,
			"xff": xff,
			"timer_support": timer_support,
			"geo_headers": geo_headers,
			"request_condition": request_condition,
		}, FastlyRequestSetting.FIELDS)
		content = self._fetch("/service/%s/version/%d/request_settings" % (service_id, version_number), method="POST", body=body)
		return FastlyRequestSetting(self, content)