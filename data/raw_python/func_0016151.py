def run(self):
		"""
		function called when thread is started
		"""
		global parallel

		if parallel:
			download_parallel(self.url, self.directory, self.idx, 
							  self.min_file_size, self.max_file_size, self.no_redirects)
		else:
			download(self.url, self.directory, self.idx,  
					 self.min_file_size, self.max_file_size, self.no_redirects)