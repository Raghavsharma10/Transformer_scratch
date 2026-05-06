def send_sms_message(sms_message, backend=None, fail_silently=False):
	"""
	Send an SMSMessage instance using a connection given by the specified `backend`.
	"""
	with get_sms_connection(backend=backend, fail_silently=fail_silently) as connection:
		result = connection.send_messages([sms_message])
	return result