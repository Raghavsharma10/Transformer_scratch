def get_page(site, page=1, **criterias):
	'Returns a paginator object and a requested page from it.'
	global _since_formats_vary

	if 'since' in criterias:
		since = criterias['since']
		if since in _since_offsets:
			since = datetime.today() - timedelta(_since_offsets[since])
		else:
			if _since_formats_vary:
				for fmt, substs in it.product( list(_since_formats),
						it.chain.from_iterable(
							it.combinations(_since_formats_vary, n)
							for n in xrange(1, len(_since_formats_vary)) ) ):
					for src, dst in substs: fmt = fmt.replace(src, dst)
					_since_formats.add(fmt)
				_since_formats_vary = None # to avoid doing it again
			for fmt in _since_formats:
				try: since = datetime.strptime(since, fmt)
				except ValueError: pass
				else: break
			else: raise Http404 # invalid format
		try:
			criterias['since'] = timezone.make_aware(
				since, timezone.get_current_timezone() )
		except (
				timezone.pytz.exceptions.AmbiguousTimeError
				if timezone.pytz else RuntimeError ):
			# Since there's no "right" way here anyway...
			criterias['since'] = since.replace(tzinfo=timezone)
	order_force = criterias.pop('asc', None)

	posts = models.Post.objects.filtered(site, **criterias)\
		.sorted(site.order_posts_by, force=order_force)\
		.select_related('feed')

	paginator = Paginator(posts, site.posts_per_page)
	try: return paginator.page(page)
	except InvalidPage: raise Http404