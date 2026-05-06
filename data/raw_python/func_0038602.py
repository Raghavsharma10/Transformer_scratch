def open_url(self, url, stale_after, parse_as_html = True, **kwargs):
		"""
		Download or retrieve from cache.

		url -- The URL to be downloaded, as a string.

		stale_after -- A network request for the url will be performed if the
		cached copy does not exist or if it exists but its age (in days) is
		larger or equal to the stale_after value. A non-positive value will
		force re-download.

		parse_as_html -- Parse the resource downloaded as HTML. This uses the
		lxml.html package to parse the resource leniently, thus it will not
		fail even for reasonably invalid HTML. This argument also decides the
		return type of this method; if True, then the return type is an
		ElementTree.Element root object; if False, the content of the resource
		is returned as a bytestring.

		Exceptions raised:

		BannedException -- If does_show_ban returns True.

		HTTPCodeNotOKError -- If the returned HTTP status code 
							  is not equal to 200.

		"""
		_LOGGER.info('open_url() received url: %s', url)
		today = datetime.date.today()
		threshold_date = today - datetime.timedelta(stale_after)
		downloaded = False

		with self._get_conn() as conn:
			rs = conn.execute('''
				select content
				from cache
				where url = ?
				and date > ?
				''',
				(url, _date_to_sqlite_str(threshold_date))
			)

		row = rs.fetchone()

		retry_run = kwargs.get('retry_run', False)
		assert (not retry_run) or (retry_run and row is None)
		if row is None:
			file_obj = self._download(url).get_file_obj()
			downloaded = True
		else:
			file_obj = cStringIO.StringIO(zlib.decompress(row[0]))

		if parse_as_html:
			tree = lxml.html.parse(file_obj)
			tree.getroot().url = url
			appears_to_be_banned = False
			if self.does_show_ban(tree.getroot()):
				appears_to_be_banned = True
				if downloaded:
					message = ('Function {f} claims we have been banned, '
							   'it was called with an element parsed from url '
							   '(downloaded, not from cache): {u}'
							   .format(f = self.does_show_ban, u = url))
					_LOGGER.error(message)
				_LOGGER.info('Deleting url %s from the cache (if it exists) '
							'because it triggered ban page cache poisoning '
							'exception', url)
				with self._get_conn() as conn:
					conn.execute('delete from cache where url = ?', [str(url)])
				if downloaded:
					raise BannedException(message)
				else:
					return self.open_url(url, stale_after, retry_run = True)
		else:
			tree = file_obj.read()

		if downloaded:
# make_links_absolute should only be called when the document has a base_url
# attribute, which it has not when it has been loaded from the database. So,
# this "if" is needed:
			if parse_as_html:
				tree.getroot().make_links_absolute(tree.getroot().base_url)
				to_store = lxml.html.tostring(
								tree,
								pretty_print = True,
								encoding = 'utf-8'
				)
			else:
				to_store = tree
			to_store = zlib.compress(to_store, 8)

			with self._get_conn() as conn:
				conn.execute('''
					insert or replace 
					into cache
					(url, date, content)
					values
					(?, ?, ?)
					''',
					(
						str(url),
						_date_to_sqlite_str(today),
						sqlite3.Binary(to_store)
					)

				)
		return tree