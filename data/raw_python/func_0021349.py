def respond_list_directory(self, dir_path, query=None):
		"""
		Respond to the client with an HTML page listing the contents of
		the specified directory.

		:param str dir_path: The path of the directory to list the contents of.
		"""
		del query
		try:
			dir_contents = os.listdir(dir_path)
		except os.error:
			self.respond_not_found()
			return
		if os.path.normpath(dir_path) != self.__config['serve_files_root']:
			dir_contents.append('..')
		dir_contents.sort(key=lambda a: a.lower())
		displaypath = html.escape(urllib.parse.unquote(self.path), quote=True)

		f = io.BytesIO()
		encoding = sys.getfilesystemencoding()
		f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n')
		f.write(b'<html>\n<title>Directory listing for ' + displaypath.encode(encoding) + b'</title>\n')
		f.write(b'<body>\n<h2>Directory listing for ' + displaypath.encode(encoding) + b'</h2>\n')
		f.write(b'<hr>\n<ul>\n')
		for name in dir_contents:
			fullname = os.path.join(dir_path, name)
			displayname = linkname = name
			# Append / for directories or @ for symbolic links
			if os.path.isdir(fullname):
				displayname = name + "/"
				linkname = name + "/"
			if os.path.islink(fullname):
				displayname = name + "@"
				# Note: a link to a directory displays with @ and links with /
			f.write(('<li><a href="' + urllib.parse.quote(linkname) + '">' + html.escape(displayname, quote=True) + '</a>\n').encode(encoding))
		f.write(b'</ul>\n<hr>\n</body>\n</html>\n')
		length = f.tell()
		f.seek(0)

		self.send_response(200)
		self.send_header('Content-Type', 'text/html; charset=' + encoding)
		self.send_header('Content-Length', length)
		self.end_headers()
		shutil.copyfileobj(f, self.wfile)
		f.close()
		return