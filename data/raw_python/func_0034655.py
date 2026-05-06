def blogroll(request, btype):
	'View that handles the generation of blogrolls.'
	response, site, cachekey = initview(request)
	if response: return response[0]

	template = loader.get_template('feedjack/{0}.xml'.format(btype))
	ctx = dict()
	fjlib.get_extra_context(site, ctx)
	ctx = Context(ctx)
	response = HttpResponse(
		template.render(ctx), content_type='text/xml; charset=utf-8' )

	patch_vary_headers(response, ['Host'])
	fjcache.cache_set(site, cachekey, (response, ctx_get(ctx, 'last_modified')))
	return response