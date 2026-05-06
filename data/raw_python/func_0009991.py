def get_rendered_page_source(self, dom_idle_requirement_secs=3, max_wait_timeout=30):
		'''
		Get the HTML markup for the current page.

		This is done by looking up the root DOM node, and then requesting the outer HTML
		for that node ID.

		This calls return will reflect any modifications made by javascript to the
		page. For unmodified content, use `blocking_navigate_and_get_source()`

		dom_idle_requirement_secs specifies the period of time for which there must have been no
		DOM modifications before treating the rendered output as "final". This call will therefore block for
		at least dom_idle_requirement_secs seconds.
		'''

		# There are a bunch of events which generally indicate a page is still doing *things*.
		# I have some concern about how this will handle things like advertisements, which
		# basically load crap forever. That's why we have the max_wait_timeout.
		target_events = [
			"Page.frameResized",
			"Page.frameStartedLoading",
			"Page.frameNavigated",
			"Page.frameAttached",
			"Page.frameStoppedLoading",
			"Page.frameScheduledNavigation",
			"Page.domContentEventFired",
			"Page.frameClearedScheduledNavigation",
			"Page.loadEventFired",
			"DOM.documentUpdated",
			"DOM.childNodeInserted",
			"DOM.childNodeRemoved",
			"DOM.childNodeCountUpdated",
		]

		start_time = time.time()
		try:
			while 1:
				if time.time() - start_time > max_wait_timeout:
					self.log.debug("Page was not idle after waiting %s seconds. Giving up and extracting content now.", max_wait_timeout)
				self.transport.recv_filtered(filter_funcs.wait_for_methods(target_events),
					tab_key=self.tab_id, timeout=dom_idle_requirement_secs)

		except ChromeResponseNotReceived:
			# We timed out, the DOM is probably idle.
			pass



		# We have to find the DOM root node ID
		dom_attr = self.DOM_getDocument(depth=-1, pierce=False)
		assert 'result' in dom_attr
		assert 'root' in dom_attr['result']
		assert 'nodeId' in dom_attr['result']['root']

		# Now, we have the root node ID.
		root_node_id = dom_attr['result']['root']['nodeId']

		# Use that to get the HTML for the specified node
		response = self.DOM_getOuterHTML(nodeId=root_node_id)

		assert 'result' in response
		assert 'outerHTML' in response['result']
		return response['result']['outerHTML']