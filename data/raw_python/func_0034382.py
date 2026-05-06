def getcloud(site, feed_id=None):
	""" Returns the tag cloud for a site or a site's subscriber.
	"""

	cloudict = fjcache.cache_get(site.id, 'tagclouds')
	if not cloudict:
		cloudict = cloudata(site)
		fjcache.cache_set(site, 'tagclouds', cloudict)

	# A subscriber's tag cloud has been requested.
	if feed_id:
		feed_id = int(feed_id)
		if feed_id in cloudict:
			return cloudict[feed_id]
		return []
	# The site tagcloud has been requested.
	return cloudict[0]