def build(site, tagdata):
	""" Returns the tag cloud for a list of tags.
	"""

	tagdata.sort()

	# we get the most popular tag to calculate the tags' weigth
	tagmax = 0
	for tagname, tagcount in tagdata:
		if tagcount > tagmax:
			tagmax = tagcount
	steps = getsteps(site.tagcloud_levels, tagmax)

	tags = []
	for tagname, tagcount in tagdata:
		weight = [twt[0] \
			for twt in steps if twt[1] >= tagcount and twt[1] > 0][0]+1
		tags.append({'tagname':tagname, 'count':tagcount, 'weight':weight})
	return tags