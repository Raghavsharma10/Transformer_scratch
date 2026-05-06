def getkey(stype, site_id=None, key=None):
	'Returns the cache key depending on its type.'
	base = '{0}.feedjack'.format(settings.CACHE_MIDDLEWARE_KEY_PREFIX)
	if stype == T_HOST: return '{0}.hostcache'.format(base)
	elif stype == T_ITEM: return '{0}.{1}.item.{2}'.format(base, site_id, str2md5(key))
	elif stype == T_META: return '{0}.{1}.meta'.format(base, site_id)
	elif stype == T_INTERVAL: return '{0}.interval.{1}'.format(base, str2md5(key))