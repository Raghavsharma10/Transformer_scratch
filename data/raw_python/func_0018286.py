def send_mass_sms(datatuple, fail_silently=False, auth_user=None, auth_password=None, connection=None):
	"""
	Given a datatuple of (subject, message, from_email, recipient_list), sends
	each message to each recipient list. Returns the number of emails sent.

	If from_email is None, the DEFAULT_FROM_EMAIL setting is used.
	If auth_user and auth_password are set, they're used to log in.
	If auth_user is None, the EMAIL_HOST_USER setting is used.
	If auth_password is None, the EMAIL_HOST_PASSWORD setting is used.

	Note: The API for this method is frozen. New code wanting to extend the
	functionality should use the EmailMessage class directly.

	https://github.com/django/django/blob/master/django/core/mail/__init__.py#L64
	"""
	import smsish.sms.backends.rq
	if isinstance(connection, smsish.sms.backends.rq.SMSBackend):
		raise NotImplementedError
	connection = connection or get_sms_connection(username=auth_user, password=auth_password, fail_silently=fail_silently)
	messages = [SMSMessage(message, from_number, recipient, connection=connection) for message, from_number, recipient in datatuple]
	return connection.send_messages(messages)