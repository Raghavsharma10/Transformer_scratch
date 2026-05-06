def mainview(request, **criterias):
	'View that handles all page requests.'
	view_data = initview(request)
	wrap = lambda func: ft.partial(func, _view_data=view_data, **criterias)
	return condition(
			etag_func=wrap(cache_etag),
			last_modified_func=wrap(cache_last_modified) )\
		(_mainview)(request, view_data, **criterias)