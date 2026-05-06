def cache_last_modified(request, *argz, **kwz):
	'''Last modification date for a cached page.
		Intended for usage in conditional views (@condition decorator).'''
	response, site, cachekey = kwz.get('_view_data') or initview(request)
	if not response: return None
	return response[1]