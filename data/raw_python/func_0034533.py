def process_entry(self, entry):
		'Construct a Post from a feedparser entry and save/update it in db'

		from feedjack.models import Post, Tag

		## Construct a Post object from feedparser entry (FeedParserDict)
		post = Post(feed=self.feed)
		post.link = entry.get('link', self.feed.link)
		post.title = entry.get('title', post.link)
		post.guid = self._get_guid(entry)

		if 'author_detail' in entry:
			post.author = entry.author_detail.get('name', '')
			post.author_email = entry.author_detail.get('email', '')
		if not post.author: post.author = entry.get('author', entry.get('creator', ''))
		if not post.author_email: post.author_email = 'nospam@nospam.com'

		try: post.content = entry.content[0].value
		except: post.content = entry.get('summary', entry.get('description', ''))

		# Try to get the post date from "updated" then "published" then "created"
		ts_parsed = ts_raw = None
		for k in self.post_timestamp_keys:
			try:
				post.date_modified = get_modified_date(
					entry.get('{0}_parsed'.format(k)), entry.get(k) )
			except ValueError as err:
				log.warn( 'Failed to process post timestamp:'
					' {0} (feed_id: {1}, post_guid: {2})'.format(err, self.feed.id, post.guid) )
			if post.date_modified: break

		post.comments = entry.get('comments', '')

		enclosures = entry.get('enclosures', list())
		if 'media_content' in entry:
			for mc in entry.media_content:
				if 'url' in mc: e = dict(href=mc['url'], medium=mc.get('medium', 'image'))
				else: e = entry.media_content
				e['type'] = 'application/x-media-content' # special ct for these things
				enclosures.append(e)
			assert enclosures, enclosures
		post.enclosures = enclosures

		## Get a list of tag objects from an entry
		# Note that these objects can't go into m2m field until properly saved
		fcat = list()
		if entry.has_key('tags'):
			for tcat in entry.tags:
				qcat = tcat.label if tcat.label is not None else tcat.term
				if not qcat: continue

				qcat = qcat.strip()
				if ',' in qcat or '/' in qcat: qcat = qcat.replace(',', '/').split('/')
				else: qcat = [qcat]

				for zcat in qcat:
					tagname = ' '.join(zcat.lower().split()).strip()[:255]
					if not tagname: continue
					if not Tag.objects.filter(name=tagname):
						cobj = Tag(name=tagname)
						cobj.save()
					fcat.append(Tag.objects.get(name=tagname))

		## Some feedback
		post_base_fields = 'title link guid author author_email'.split()

		log.debug('[{0}] Entry\n{1}'.format(self.feed.id, '\n'.join(
			['  {0}: {1}'.format(key, getattr(post, key)) for key in post_base_fields]
			+ ['tags: {0}'.format(' '.join(it.imap(op.attrgetter('name'), fcat)))] )))

		## Store / update a post
		if post.guid in self.postdict: # post exists, update if it was modified (and feed is mutable)
			post_old = self.postdict[post.guid]
			changed = post_old.content != post.content or (
				post.date_modified and post_old.date_modified != post.date_modified )

			if not self.feed.immutable and changed:
				retval = ENTRY_UPDATED
				log.extra('[{0}] Updating existing post: {1}'.format(self.feed.id, post.link))
				# Update fields
				for field in post_base_fields + ['content', 'comments']:
					setattr(post_old, field, getattr(post, field))
				post_old.date_modified = post.date_modified or post_old.date_modified
				# Update tags
				post_old.tags.clear()
				for tcat in fcat: post_old.tags.add(tcat)
				post_old.save()
			else:
				retval = ENTRY_SAME
				log.extra( ( '[{0}] Post has not changed: {1}' if not changed else
					'[{0}] Post changed, but feed is marked as immutable: {1}' )\
						.format(self.feed.id, post.link) )

		else: # new post, store it into database
			retval = ENTRY_NEW
			log.extra( '[{0}] Saving new post: {1} (timestamp: {2})'\
				.format(self.feed.id, post.guid, post.date_modified) )

			# Try hard to set date_modified: feed.modified, http.modified and now() as a last resort
			if not post.date_modified and self.fpf:
				try:
					post.date_modified = get_modified_date(
						self.fpf.feed.get('modified_parsed') or self.fpf.get('modified_parsed'),
						self.fpf.feed.get('modified') or self.fpf.get('modified') )
				except ValueError as err:
					log.warn(( 'Failed to process feed/http timestamp: {0} (feed_id: {1},'
						' post_guid: {2}), falling back to "now"' ).format(err, self.feed.id, post.guid))
				if not post.date_modified:
					post.date_modified = timezone.now()
					log.debug(( '[{0}] Using current time for post'
						' ({1}) timestamp' ).format(self.feed.id, post.guid))
				else:
					log.debug(
						'[{0}] Using timestamp from feed/http for post ({1}): {2}'\
						.format(self.feed.id, post.guid, post.date_modified) )

			if self.options.hidden: post.hidden = True
			try: post.save()
			except IntegrityError:
				log.error( 'IntegrityError while saving (supposedly) new'\
					' post with guid: {0.guid}, link: {0.link}, title: {0.title}'.format(post) )
				raise
			for tcat in fcat: post.tags.add(tcat)
			self.postdict[post.guid] = post

		return retval