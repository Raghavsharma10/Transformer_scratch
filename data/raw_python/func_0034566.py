def pick_enclosure_link(post, parameter=''):
	'''Override URL of the Post to point to url of the first enclosure with
			href attribute non-empty and type matching specified regexp parameter (empty=any).
		Missing "type" attribute for enclosure will be matched as an empty string.
		If none of the enclosures match, link won't be updated.'''
	for e in (post.enclosures or list()):
		href = e.get('href')
		if not href: continue
		if parameter and not re.search(parameter, e.get('type', '')): continue
		return dict(link=href)