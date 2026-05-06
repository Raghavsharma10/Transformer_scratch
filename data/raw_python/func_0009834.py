def close_websockets(self):
		""" Close websocket connection to remote browser."""
		self.log.info("Websocket Teardown called")
		for key in list(self.soclist.keys()):
			if self.soclist[key]:
				self.soclist[key].close()
			self.soclist.pop(key)