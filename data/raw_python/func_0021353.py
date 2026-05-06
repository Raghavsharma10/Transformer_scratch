def dispatch_handler(self, query=None):
		"""
		Dispatch functions based on the established handler_map. It is
		generally not necessary to override this function and doing so
		will prevent any handlers from being executed. This function is
		executed automatically when requests of either GET, HEAD, or POST
		are received.

		:param dict query: Parsed query parameters from the corresponding request.
		"""
		query = (query or {})
		# normalize the path
		# abandon query parameters
		self.path = self.path.split('?', 1)[0]
		self.path = self.path.split('#', 1)[0]
		original_path = urllib.parse.unquote(self.path)
		self.path = posixpath.normpath(original_path)
		words = self.path.split('/')
		words = filter(None, words)
		tmp_path = ''
		for word in words:
			_, word = os.path.splitdrive(word)
			_, word = os.path.split(word)
			if word in (os.curdir, os.pardir):
				continue
			tmp_path = os.path.join(tmp_path, word)
		self.path = tmp_path

		if self.path == 'robots.txt' and self.__config['serve_robots_txt']:
			self.send_response_full(self.__config['robots_txt'])
			return

		self.cookies = http.cookies.SimpleCookie(self.headers.get('cookie', ''))
		handler, is_method = self.__get_handler(is_rpc=False)
		if handler is not None:
			try:
				handler(*((query,) if is_method else (self, query)))
			except Exception:
				self.respond_server_error()
			return

		if not self.__config['serve_files']:
			self.respond_not_found()
			return

		file_path = self.__config['serve_files_root']
		file_path = os.path.join(file_path, tmp_path)
		if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
			self.respond_file(file_path, query=query)
			return
		elif os.path.isdir(file_path) and os.access(file_path, os.R_OK):
			if not original_path.endswith('/'):
				# redirect browser, doing what apache does
				destination = self.path + '/'
				if self.command == 'GET' and self.query_data:
					destination += '?' + urllib.parse.urlencode(self.query_data, True)
				self.respond_redirect(destination)
				return
			for index in ['index.html', 'index.htm']:
				index = os.path.join(file_path, index)
				if os.path.isfile(index) and os.access(index, os.R_OK):
					self.respond_file(index, query=query)
					return
			if self.__config['serve_files_list_directories']:
				self.respond_list_directory(file_path, query=query)
				return
		self.respond_not_found()
		return