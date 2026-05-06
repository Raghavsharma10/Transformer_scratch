def initview(request, response_cache=True):
	'''Retrieves the basic data needed by all feeds (host, feeds, etc)
		Returns a tuple of:
			1. A valid cached response or None
			2. The current site object
			3. The cache key
			4. The subscribers for the site (objects)
			5. The feeds for the site (ids)'''

	http_host, path_info = ( smart_unicode(part.strip('/')) for part in
		[ request.META['HTTP_HOST'],
			request.META.get('REQUEST_URI', request.META.get('PATH_INFO', '/')) ] )
	query_string = request.META['QUERY_STRING']

	url = '{0}/{1}'.format(http_host, path_info)
	cachekey = u'{0}?{1}'.format(*it.imap(smart_unicode, (path_info, query_string)))
	hostdict = fjcache.hostcache_get() or dict()

	site_id = hostdict[url] if url in hostdict else None
	if site_id and response_cache:
		response = fjcache.cache_get(site_id, cachekey)
		if response: return response, None, cachekey

	if site_id: site = models.Site.objects.get(pk=site_id)
	else: # match site from all of them
		sites = list(models.Site.objects.all())

		if not sites:
			# Somebody is requesting something, but the user
			#  didn't create a site yet. Creating a default one...
			site = models.Site(
				name='Default Feedjack Site/Planet',
				url=request.build_absolute_uri(request.path),
				title='Feedjack Site Title',
				description='Feedjack Site Description.'
					' Please change this in the admin interface.',
				template='bootstrap' )
			site.save()

		else:
			# Select the most matching site possible,
			#  preferring "default" when everything else is equal
			results = defaultdict(list)
			for site in sites:
				relevance, site_url = 0, urlparse(site.url)
				if site_url.netloc == http_host: relevance += 10 # host matches
				if path_info.startswith(site_url.path.strip('/')): relevance += 10 # path matches
				if site.default_site: relevance += 5 # marked as "default"
				results[relevance].append((site_url, site))
			for relevance in sorted(results, reverse=True):
				try: site_url, site = results[relevance][0]
				except IndexError: pass
				else: break
			if site_url.netloc != http_host: # redirect to proper site hostname
				response = HttpResponsePermanentRedirect(
					'http://{0}/{1}{2}'.format( site_url.netloc, path_info,
						'?{0}'.format(query_string) if query_string.strip() else '') )
				return (response, timezone.now()), None, cachekey

		hostdict[url] = site_id = site.id
		fjcache.hostcache_set(hostdict)

	if response_cache:
		response = fjcache.cache_get(site_id, cachekey)
		if response: return response, None, cachekey

	return None, site, cachekey