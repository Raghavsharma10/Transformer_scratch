def feed_interval_set(feed_id, parameters, interval, interval_ts):
	'Set adaptive interval between checks for a feed.'
	cache.set(getkey( T_INTERVAL,
		key=feed_interval_key(feed_id, parameters) ), (interval, interval_ts))