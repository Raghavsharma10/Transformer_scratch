def page_context(request, site, **criterias):
	'Returns the context dictionary for a page view.'
	try: page = int(request.GET.get('page', 1))
	except ValueError: page = 1

	feed, tag = criterias.get('feed'), criterias.get('tag')
	if feed:
		try: feed = models.Feed.objects.get(pk=feed)
		except ObjectDoesNotExist: raise Http404

	page = get_page(site, page=page, **criterias)
	subscribers = site.active_subscribers

	if site.show_tagcloud and page.object_list:
		from feedjack import fjcloud
		# This will hit the DB once per page instead of once for every post in
		#  a page. To take advantage of this the template designer must call
		#  the qtags property in every item, instead of the default tags property.
		user_obj, tag_obj = get_posts_tags(
			subscribers, page.object_list, feed, tag )
		tag_cloud = fjcloud.getcloud(site, feed and feed.id)
	else:
		tag_obj, tag_cloud = None, tuple()
		try:
			user_obj = models.Subscriber.objects\
				.get(site=site, feed=feed) if feed else None
		except ObjectDoesNotExist: raise Http404

	site_proc_tags = site.processing_tags.strip()
	if site_proc_tags != 'none':
		site_proc_tags = filter( None,
			map(op.methodcaller('strip'), site.processing_tags.split(',')) )
		# XXX: database hit that can be cached
		for site_feed, posts in it.groupby(page.object_list, key=op.attrgetter('feed')):
			proc = site_feed.processor_for_tags(site_proc_tags)
			if proc: proc.apply_overlay_to_posts(posts)

	ctx = dict(
		last_modified = max(it.imap(
				op.attrgetter('date_updated'), page.object_list ))\
			if len(page.object_list) else datetime(1970, 1, 1, 0, 0, 0, 0, timezone.utc),

		object_list = page.object_list,
		subscribers = subscribers.select_related('feed'),
		tag = tag_obj,
		tagcloud = tag_cloud,

		feed = feed,
		url_suffix = ''.join((
			'/feed/{0}'.format(feed.id) if feed else '',
			'/tag/{0}'.format(escape(tag)) if tag else '' )),

		p = page, # "page" is taken by legacy number
		p_10neighbors = OrderedDict(
			# OrderedDict of "num: exists" values
			# Use as "{% for p_num, p_exists in p_10neighbors.items|slice:"7:-7" %}"
			(p, p >= 1 and p <= page.paginator.num_pages)
			for p in ((page.number + n) for n in xrange(-10, 11)) ),

		## DEPRECATED:

		# Totally misnamed and inconsistent b/w user/user_obj,
		#  use "feed" and "subscribers" instead.
		user_id = feed and feed.id,
		user = user_obj,

		# Legacy flat pagination context, use "p" instead.
		is_paginated = page.paginator.num_pages > 1,
		results_per_page = site.posts_per_page,
		has_next = page.has_next(),
		has_previous = page.has_previous(),
		page = page.number,
		next = page.number + 1,
		previous = page.number - 1,
		pages = page.paginator.num_pages,
		hits = page.paginator.count )

	get_extra_context(site, ctx)

	return ctx