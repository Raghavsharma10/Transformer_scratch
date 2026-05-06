def similar_title(post, parameter=None):
	'''Skip posts with fuzzy-matched (threshold = levenshtein distance / length) title.
		Parameters (comma-delimited):
			minimal threshold, at which values are considired similar (float, 0 < x < 1, default: {0});
			comparison timespan, seconds (int, 0 = inf, default: {1}).'''
	from feedjack.models import Post
	threshold, timespan = DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_SIMILARITY_TIMESPAN
	if parameter:
		parameter = map(op.methodcaller('strip'), parameter.split(',', 1))
		threshold = parameter.pop()
		try: threshold, timespan = parameter.pop(), threshold
		except IndexError: pass
		threshold, timespan = float(threshold), int(timespan)
	similar = Post.objects.filtered(for_display=False)\
		.exclude(id=post.id).similar(threshold, title=post.title)
	if timespan:
		similar = similar.filter(date_updated__gt=timezone.now() - timedelta(seconds=timespan))
	return not bool(similar.exists())