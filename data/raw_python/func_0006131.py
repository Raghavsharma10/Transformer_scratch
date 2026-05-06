def download(self):
		''' Downloads HTML from url. '''

		self.page = requests.get(self.url)
		self.tree = html.fromstring(self.page.text)