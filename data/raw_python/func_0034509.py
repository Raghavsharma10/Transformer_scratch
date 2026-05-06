def get_posts_tags(subscribers, object_list, feed, tag_name):
	'''Adds a qtags property in every post object in a page.
		Use "qtags" instead of "tags" in templates to avoid unnecesary DB hits.'''

	tagd = dict()
	user_obj = None
	tag_obj = None
	tags = models.Tag.objects.extra(
		select=dict(post_id='{0}.{1}'.format(
			*it.imap( connection.ops.quote_name,
				('feedjack_post_tags', 'post_id') ) )),
		tables=['feedjack_post_tags'],
		where=[
		'{0}.{1}={2}.{3}'.format(*it.imap( connection.ops.quote_name,
			('feedjack_tag', 'id', 'feedjack_post_tags', 'tag_id') )),
		'{0}.{1} IN ({2})'.format(
			connection.ops.quote_name('feedjack_post_tags'),
			connection.ops.quote_name('post_id'),
			', '.join([str(post.id) for post in object_list]) ) ] )

	for tag in tags:
		if tag.post_id not in tagd: tagd[tag.post_id] = list()
		tagd[tag.post_id].append(tag)
		if tag_name and tag.name == tag_name: tag_obj = tag

	subd = dict()
	for sub in subscribers: subd[sub.feed.id] = sub
	for post in object_list:
		if post.id in tagd: post.qtags = tagd[post.id]
		else: post.qtags = list()
		post.subscriber = subd[post.feed.id]
		if feed == post.feed: user_obj = post.subscriber

	return user_obj, tag_obj