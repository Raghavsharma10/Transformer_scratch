def buildfeed(request, feedclass, **criterias):
	'View that handles the feeds.'
	view_data = initview(request)
	wrap = lambda func: ft.partial(func, _view_data=view_data, **criterias)
	return condition(
			etag_func=wrap(cache_etag),
			last_modified_func=wrap(cache_last_modified) )\
		(_buildfeed)(request, feedclass, view_data, **criterias)