def blocking_navigate(self, url, timeout=DEFAULT_TIMEOUT_SECS):
		'''
		Do a blocking navigate to url `url`.

		This function triggers a navigation, and then waits for the browser
		to claim the page has finished loading.

		Roughly, this corresponds to the javascript `DOMContentLoaded` event,
		meaning the dom for the page is ready.


		Internals:

		A navigation command results in a sequence of events:

		 - Page.frameStartedLoading" (with frameid)
		 - Page.frameStoppedLoading" (with frameid)
		 - Page.loadEventFired" (not attached to an ID)

		Therefore, this call triggers a navigation option,
		and then waits for the expected set of response event messages.

		'''

		self.transport.flush(tab_key=self.tab_id)

		ret = self.Page_navigate(url = url)

		assert("result"   in ret),           "Missing return content"
		assert("frameId"  in ret['result']), "Missing 'frameId' in return content"
		assert("loaderId" in ret['result']), "Missing 'loaderId' in return content"

		expected_id = ret['result']['frameId']
		loader_id   = ret['result']['loaderId']

		try:
			self.log.debug("Waiting for frame navigated command response.")
			self.transport.recv_filtered(filter_funcs.check_frame_navigated_command(expected_id), tab_key=self.tab_id, timeout=timeout)
			self.log.debug("Waiting for frameStartedLoading response.")
			self.transport.recv_filtered(filter_funcs.check_frame_load_command("Page.frameStartedLoading"), tab_key=self.tab_id, timeout=timeout)
			self.log.debug("Waiting for frameStoppedLoading response.")
			self.transport.recv_filtered(filter_funcs.check_frame_load_command("Page.frameStoppedLoading"), tab_key=self.tab_id, timeout=timeout)
			# self.transport.recv_filtered(check_load_event_fired, tab_key=self.tab_id, timeout=timeout)

			self.log.debug("Waiting for responseReceived response.")
			resp = self.transport.recv_filtered(filter_funcs.network_response_recieved_for_url(url=None, expected_id=expected_id), tab_key=self.tab_id, timeout=timeout)

			if resp is None:
				raise ChromeNavigateTimedOut("Blocking navigate timed out!")

			return resp['params']
		# The `Page.frameNavigated ` event does not get fired for non-markup responses.
		# Therefore, if we timeout on waiting for that, check to see if we received a binary response.
		except ChromeResponseNotReceived:
			# So this is basically broken, fix is https://bugs.chromium.org/p/chromium/issues/detail?id=831887
			# but that bug report isn't fixed yet.
			# Siiiigh.
			self.log.warning("Failed to receive expected response to navigate command. Checking if response is a binary object.")
			resp = self.transport.recv_filtered(
				keycheck = filter_funcs.check_frame_loader_command(
						method_name = "Network.responseReceived",
						loader_id   = loader_id
					),
				tab_key  = self.tab_id,
				timeout  = timeout)

			return resp['params']