def same_guid(post, parameter=DEFAULT_SIMILARITY_TIMESPAN):
	'''Skip posts with exactly same GUID.
		Parameter: comparison timespan, seconds (int, 0 = inf, default: {0}).'''
	from feedjack.models import Post
	if isinstance(parameter, types.StringTypes): parameter = int(parameter.strip())
	similar = Post.objects.filtered(for_display=False)\
		.exclude(id=post.id).filter(guid=post.guid)
	if parameter:
		similar = similar.filter(date_updated__gt=timezone.now() - timedelta(seconds=parameter))
	return not bool(similar.exists())