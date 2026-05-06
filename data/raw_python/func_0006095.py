def get_message_dict(self):
        """
        Generate the various parts of the message and return them in a
        dictionary, suitable for passing directly as keyword arguments
        to ``django.core.mail.send_mail()``.

        By default, the following values are returned:

        * ``from_email``

        * ``message``

        * ``recipient_list``

        * ``subject``

        """
        if not self.is_valid():
            raise ValueError(
                "Message cannot be sent from invalid contact form"
            )
        message_dict = {}
        for message_part in ('from_email', 'message',
                             'recipient_list', 'subject'):
            attr = getattr(self, message_part)
            message_dict[message_part] = attr() if callable(attr) else attr
        return message_dict