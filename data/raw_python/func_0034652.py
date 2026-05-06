def cache_etag(request, *argz, **kwz):
	'''Produce etag value for a cached page.
		Intended for usage in conditional views (@condition decorator).'''
	response, site, cachekey = kwz.get('_view_data') or initview(request)
	if not response: return None
	return fjcache.str2md5(
		'{0}--{1}--{2}'.format( site.id if site else 'x', cachekey,
			response[1].strftime('%Y-%m-%d %H:%M:%S%z') ) )