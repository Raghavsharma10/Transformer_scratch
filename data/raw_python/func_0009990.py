def blocking_navigate_and_get_source(self, url, timeout=DEFAULT_TIMEOUT_SECS):
		'''
		Do a blocking navigate to url `url`, and then extract the
		response body and return that.

		This effectively returns the *unrendered* page content that's sent over the wire. As such,
		if the page does any modification of the contained markup during rendering (via javascript), this
		function will not reflect the changes made by the javascript.

		The rendered page content can be retreived by calling `get_rendered_page_source()`.

		Due to the remote api structure, accessing the raw content after the content has been loaded
		is not possible, so any task requiring the raw content must be careful to request it
		before it actually navigates to said content.

		Return value is a dictionary with two keys:
		{
			'binary' : (boolean, true if content is binary, false if not)
			'content' : (string of bytestring, depending on whether `binary` is true or not)
		}

		'''


		resp = self.blocking_navigate(url, timeout)
		assert 'requestId' in resp
		assert 'response' in resp
		# self.log.debug('blocking_navigate Response %s', pprint.pformat(resp))

		ctype = 'application/unknown'

		resp_response = resp['response']

		if 'mimeType' in resp_response:
			ctype = resp_response['mimeType']
		if 'headers' in resp_response and 'content-type' in resp_response['headers']:
			ctype = resp_response['headers']['content-type'].split(";")[0]

		self.log.debug("Trying to get response body")
		try:
			ret = self.get_unpacked_response_body(resp['requestId'], mimetype=ctype)
		except ChromeError:
			ret = self.handle_page_location_changed(timeout)

		return ret