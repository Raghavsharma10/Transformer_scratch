def hash(value, chars=None):
	'Get N chars (default: all) of secure hash hexdigest of value.'
	value = hash_func(value).hexdigest()
	if chars: value = value[:chars]
	return mark_safe(value)