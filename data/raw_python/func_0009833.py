def connect(self, tab_key):
		"""
		Open a websocket connection to remote browser, determined by
		self.host and self.port.  Each tab has it's own websocket
		endpoint.

		"""

		assert self.tablist is not None

		tab_idx = self._get_tab_idx_for_key(tab_key)

		if not self.tablist:
			self.tablist = self.fetch_tablist()

		for fails in range(9999):
			try:
				# If we're one past the end of the tablist, we need to create a new tab
				if tab_idx is None:
					self.log.debug("Creating new tab (%s active)", len(self.tablist))
					self.__create_new_tab(tab_key)

				self.__connect_to_tab(tab_key)
				break

			except cr_exceptions.ChromeConnectFailure as e:
				if fails > 6:
					self.log.error("Failed to fetch tab websocket URL after %s retries. Aborting!", fails)
					raise e
				self.log.info("Tab may not have started yet (%s tabs active). Recreating.", len(self.tablist))
				# self.log.info("Tag: %s", self.tablist[tab_idx])


				# For reasons I don't understand, sometimes a new tab doesn't get a websocket
				# debugger URL. Anyways, we close and re-open the tab if that happens.
				# TODO: Handle the case when this happens on the first tab. I think closing the first
				#       tab will kill chromium.
				self.__close_tab(tab_key)