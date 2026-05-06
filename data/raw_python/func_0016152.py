def start(self):
		"""
		function to initialize thread for downloading
		"""
		global parallel
		for self.i in range(0, self.length):
			if parallel:
				self.thread.append(myThread(self.url[ self.i ], self.directory, self.i, 
								   self.min_file_size, self.max_file_size, self.no_redirects))
			else:
				# if not parallel whole url list is passed
				self.thread.append(myThread(self.url, self.directory, self.i , self.min_file_size, 
								   self.max_file_size,  self.no_redirects))
			self.progress[self.i]["value"] = 0
			self.bytes[self.i] = 0
			self.thread[self.i].start()

		self.read_bytes()