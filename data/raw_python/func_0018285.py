def send_sms(message, from_number, recipient_list, fail_silently=False, auth_user=None, auth_password=None, connection=None):
	"""
	Easy wrapper for sending a single message to a recipient list. All members
	of the recipient list will see the other recipients in the 'To' field.

	If auth_user is None, the EMAIL_HOST_USER setting is used.
	If auth_password is None, the EMAIL_HOST_PASSWORD setting is used.

	Note: The API for this method is frozen. New code wanting to extend the
	functionality should use the EmailMessage class directly.

	https://github.com/django/django/blob/master/django/core/mail/__init__.py#L40
	"""
	connection = connection or get_sms_connection(username=auth_user, password=auth_password, fail_silently=fail_silently)
	mail = SMSMessage(message, from_number, recipient_list, connection=connection)

	return mail.send()