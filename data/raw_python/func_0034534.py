def _process(self):
		'Downloads and parses a feed.'

		ret_values = {
			ENTRY_NEW: 0,
			ENTRY_UPDATED: 0,
			ENTRY_SAME: 0,
			ENTRY_ERR: 0 }
		report_errors = not self.options.report_after\
			or not self.feed.last_checked\
			or (self.feed.last_checked + self.options.report_after < timezone.now())

		feedparser_kws = dict()
		if sys.hexversion >= 0x2070900 and not self.feed.verify_tls_certs:
			import urllib2, ssl
			ctx = ssl.create_default_context()
			ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
			feedparser_kws['handlers'] = [urllib2.HTTPSHandler(context=ctx)]

		try:
			self.fpf = feedparser.parse( self.feed.feed_url, agent=USER_AGENT,
				etag=self.feed.etag if not self.options.force else '', **feedparser_kws )
		except KeyboardInterrupt: raise
		except:
			if report_errors:
				log.error( 'Feed cannot be parsed: {0} (#{1})'\
					.format(self.feed.feed_url, self.feed.id) )
			return FEED_ERRPARSE, ret_values

		if hasattr(self.fpf, 'status'):
			log.extra('[{0}] HTTP status {1}: {2}'.format(
				self.feed.id, self.fpf.status, self.feed.feed_url ))
			if self.fpf.status == 304:
				log.extra(( '[{0}] Feed has not changed since '
					'last check: {1}' ).format(self.feed.id, self.feed.feed_url))
				# Fast-path: just update last_checked timestamp
				self.feed.last_checked = timezone.now()
				self.feed.save()
				return FEED_SAME, ret_values

			if self.fpf.status >= 400:
				if report_errors:
					log.warn('[{0}] HTTP error {1}: {2}'.format(
						self.feed.id, self.fpf.status, self.feed.feed_url ))
				return FEED_ERRFETCH, ret_values

		if self.fpf.bozo:
			bozo = getattr(self.fpf, 'bozo_exception', 'unknown error')
			if not self.feed.skip_errors:
				if report_errors:
					log.warn( '[{0}] Failed to fetch feed: {1} ({2})'\
						.format(self.feed.id, self.feed.feed_url, bozo) )
				return FEED_ERRFETCH, ret_values
			elif report_errors:
				log.info( '[{0}] Skipped feed error: {1} ({2})'\
					.format(self.feed.id, self.feed.feed_url, bozo) )

		self.feed.title = self.fpf.feed.get('title', '')[:200]
		self.feed.tagline = self.fpf.feed.get('tagline', '')
		self.feed.link = self.fpf.feed.get('link', '')
		self.feed.last_checked = timezone.now()

		log.debug('[{0}] Feed info for: {1}\n{2}'.format(
			self.feed.id, self.feed.feed_url, '\n'.join(
			'  {0}: {1}'.format(key, getattr(self.feed, key))
			for key in ['title', 'tagline', 'link', 'last_checked'] )))

		guids = filter(None, it.imap(self._get_guid, self.fpf.entries))
		if guids:
			from feedjack.models import Post
			self.postdict = dict( (post.guid, post)
				for post in Post.objects.filter(
					feed=self.feed.id, guid__in=guids ) )
			if self.options.max_diff:
				# Do not calculate diff for empty (probably just-added) feeds
				if not self.postdict and Post.objects.filter(feed=self.feed.id).count() == 0: diff = 0
				else: diff = op.truediv(len(guids) - len(self.postdict), len(guids)) * 100
				if diff > self.options.max_diff:
					log.warn( '[{0}] Feed validation failed: {1} (diff: {2}% > {3}%)'\
						.format(self.feed.id, self.feed.feed_url, round(diff, 1), self.options.max_diff) )
					return FEED_INVALID, ret_values
		else: self.postdict = dict()

		self.feed.save() # etag/mtime aren't updated yet

		for entry in self.fpf.entries:
			try:
				with transaction.atomic(): ret_entry = self.process_entry(entry)
			except:
				print_exc(self.feed.id)
				ret_entry = ENTRY_ERR
			ret_values[ret_entry] += 1

		if not ret_values[ENTRY_ERR]: # etag/mtime updated only if there's no errors
			self.feed.etag = self.fpf.get('etag') or ''
			try: self.feed.last_modified = feedparser_ts(self.fpf.modified_parsed)
			except AttributeError: pass
			self.feed.save()

		return FEED_OK if ret_values[ENTRY_NEW]\
			or ret_values[ENTRY_UPDATED] else FEED_SAME, ret_values