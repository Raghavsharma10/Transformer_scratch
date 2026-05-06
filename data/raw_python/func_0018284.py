def get_sms_connection(backend=None, fail_silently=False, **kwds):
	"""Load an sms backend and return an instance of it.

	If backend is None (default) settings.SMS_BACKEND is used.

	Both fail_silently and other keyword arguments are used in the
	constructor of the backend.

	https://github.com/django/django/blob/master/django/core/mail/__init__.py#L28
	"""
	klass = import_string(backend or settings.SMS_BACKEND)
	return klass(fail_silently=fail_silently, **kwds)