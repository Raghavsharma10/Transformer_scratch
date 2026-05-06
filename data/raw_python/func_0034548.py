def cache_set(site, key, data):
	'''Sets cache data for a site.
		All keys related to a site are stored in a meta key. This key is per-site.'''
	tkey = getkey(T_ITEM, site.id, key)
	mkey = getkey(T_META, site.id)
	tmp = cache.get(mkey)
	longdur = 365*24*60*60
	if not tmp:
		tmp = [tkey]
		cache.set(mkey, [tkey], longdur)
	elif tkey not in tmp:
		tmp.append(tkey)
		cache.set(mkey, tmp, longdur)
	cache.set(tkey, data, site.cache_duration)