def cache_delsite(site_id):
	'Removes all cache data from a site.'
	mkey = getkey(T_META, site_id)
	tmp = cache.get(mkey)
	if not tmp:
		return
	for tkey in tmp:
		cache.delete(tkey)
	cache.delete(mkey)