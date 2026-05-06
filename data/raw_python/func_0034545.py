def feed_interval_get(feed_id, parameters):
	'Get adaptive interval between checks for a feed.'
	val = cache.get(getkey( T_INTERVAL,
		key=feed_interval_key(feed_id, parameters) ))
	return val if isinstance(val, tuple) else (val, None)