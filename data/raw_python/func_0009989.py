def handle_page_location_changed(self, timeout=None):
		'''
		If the chrome tab has internally redirected (generally because jerberscript), this
		will walk the page navigation responses and attempt to fetch the response body for
		the tab's latest location.
		'''

		# In general, this is often called after other mechanisms have confirmed
		# that the tab has already navigated. As such, we want to not wait a while
		# to discover something went wrong, so use a timeout that basically just
		# results in checking the available buffer, and nothing else.
		if not timeout:
			timeout = 0.1

		self.log.debug("We may have redirected. Checking.")

		messages = self.transport.recv_all_filtered(filter_funcs.capture_loading_events, tab_key=self.tab_id)
		if not messages:
			raise ChromeError("Couldn't track redirect! No idea what to do!")

		last_message = messages[-1]
		self.log.info("Probably a redirect! New content url: '%s'", last_message['params']['documentURL'])

		resp = self.transport.recv_filtered(filter_funcs.network_response_recieved_for_url(last_message['params']['documentURL'], last_message['params']['frameId']), tab_key=self.tab_id)
		resp = resp['params']

		ctype = 'application/unknown'

		resp_response = resp['response']

		if 'mimeType' in resp_response:
			ctype = resp_response['mimeType']
		if 'headers' in resp_response and 'content-type' in resp_response['headers']:
			ctype = resp_response['headers']['content-type'].split(";")[0]

		# We assume the last document request was the redirect.
		# This is /probably/ kind of a poor practice, but what the hell.
		# I have no idea what this would do if there are non-html documents (or if that can even happen.)
		return self.get_unpacked_response_body(last_message['params']['requestId'], mimetype=ctype)