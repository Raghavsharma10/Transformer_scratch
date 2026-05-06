def get_extra_context(site, ctx):
	'Returns extra data useful to the templates.'
	# XXX: clean this up from obsolete stuff
	ctx['site'] = site
	ctx['feeds'] = feeds = site.active_feeds.order_by('name')

	def get_mod_chk(k):
		mod, chk = (
			(max(vals) if vals else None) for vals in (
				filter(None, it.imap(op.attrgetter(k), feeds))
				for k in ['last_modified', 'last_checked'] ) )
		chk = chk or datetime(1970, 1, 1, 0, 0, 0, 0, timezone.utc)
		ctx['last_modified'], ctx['last_checked'] = mod or chk, chk
		return ctx[k]
	for k in 'last_modified', 'last_checked':
		ctx[k] = lambda: get_mod_chk(k)

	# media_url is set here for historical reasons,
	#  use static_url or STATIC_URL (from django context) in any new templates.
	ctx['media_url'] = ctx['static_url'] =\
		'{}feedjack/{}'.format(settings.STATIC_URL, site.template)