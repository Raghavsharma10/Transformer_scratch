def feed_interval_delete(feed_id, parameters):
	'Invalidate cached adaptive interval value.'
	cache.delete(getkey( T_INTERVAL,
		key=feed_interval_key(feed_id, parameters) ))