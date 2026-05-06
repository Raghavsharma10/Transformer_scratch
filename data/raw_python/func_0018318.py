def send_messages(self, messages):
		"""Redirect messages to the dummy outbox"""
		msg_count = 0
		for message in messages:  # .message() triggers header validation
			message.message()
			msg_count += 1
		mail.outbox.extend(messages)
		return msg_count