def prettyhtml(value, autoescape=None):
	'Clean (and optionally escape) passed html of unsafe tags and attributes.'
	value = html_cleaner(value)
	return escape(value) if autoescape\
		and not isinstance(value, SafeData) else mark_safe(value)