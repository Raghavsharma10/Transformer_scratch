def respond_file(self, file_path, attachment=False, query=None):
		"""
		Respond to the client by serving a file, either directly or as
		an attachment.

		:param str file_path: The path to the file to serve, this does not need to be in the web root.
		:param bool attachment: Whether to serve the file as a download by setting the Content-Disposition header.
		"""
		del query
		file_path = os.path.abspath(file_path)
		try:
			file_obj = open(file_path, 'rb')
		except IOError:
			self.respond_not_found()
			return
		self.send_response(200)
		self.send_header('Content-Type', self.guess_mime_type(file_path))
		fs = os.fstat(file_obj.fileno())
		self.send_header('Content-Length', str(fs[6]))
		if attachment:
			file_name = os.path.basename(file_path)
			self.send_header('Content-Disposition', 'attachment; filename=' + file_name)
		self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
		self.end_headers()
		shutil.copyfileobj(file_obj, self.wfile)
		file_obj.close()
		return