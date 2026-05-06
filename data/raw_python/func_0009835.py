def fetch_tablist(self):
		"""Connect to host:port and request list of tabs
			 return list of dicts of data about open tabs."""
		# find websocket endpoint
		try:
			response = requests.get("http://%s:%s/json" % (self.host, self.port))
		except requests.exceptions.ConnectionError:
			raise cr_exceptions.ChromeConnectFailure("Failed to fetch configuration json from browser!")

		tablist = json.loads(response.text)

		return tablist