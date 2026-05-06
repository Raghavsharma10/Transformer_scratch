def get_by_string(cls, fields, query):
	'''Get object by numeric id or exact
		and unique part of specified attrs (name, title, etc).'''
	try: pk = int(query)
	except ValueError: pass
	else: return cls.objects.get(pk=pk)
	obj = list(cls.objects.filter(reduce( op.or_,
		list(Q(**{'{}__icontains'.format(f): query}) for f in fields) )))
	if len(obj) > 1:
		raise cls.MultipleObjectsReturned((
			u'Unable to uniquely identify {}'
				' by provided criteria: {!r} (candidates: {})' )\
			.format(cls.__name__, query, ', '.join(it.imap(unicode, obj))) )
	elif not len(obj):
		raise cls.DoesNotExist(
			u'Unable to find site by provided criteria: {!r}'.format(query) )
	return obj[0]