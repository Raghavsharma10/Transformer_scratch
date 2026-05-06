def cloudata(site):
	""" Returns a dictionary with all the tag clouds related to a site.
	"""

	# XXX: this looks like it can be done via ORM
	tagdata = getquery("""
		SELECT feedjack_post.feed_id, feedjack_tag.name, COUNT(*)
		FROM feedjack_post, feedjack_subscriber, feedjack_tag,
		feedjack_post_tags
		WHERE feedjack_post.feed_id=feedjack_subscriber.feed_id AND
		feedjack_post_tags.tag_id=feedjack_tag.id AND
		feedjack_post_tags.post_id=feedjack_post.id AND
		feedjack_subscriber.site_id=%d
		GROUP BY feedjack_post.feed_id, feedjack_tag.name
		ORDER BY feedjack_post.feed_id, feedjack_tag.name""" % site.id)
	tagdict = {}
	globaldict = {}
	cloudict = {}
	for feed_id, tagname, tagcount in tagdata:
		if feed_id not in tagdict:
			tagdict[feed_id] = []
		tagdict[feed_id].append((tagname, tagcount))
		try:
			globaldict[tagname] += tagcount
		except KeyError:
			globaldict[tagname] = tagcount
	tagdict[0] = globaldict.items()
	for key, val in tagdict.items():
		cloudict[key] = build(site, val)
	return cloudict