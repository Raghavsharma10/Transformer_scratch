def request(self, subject, callback, msg=None):
        """
        ublish a message with an implicit inbox listener as the reply.
        Message is optional.

        Args:
            subject (string): a string with the subject
            callback (function): callback to be called
            msg (string=None): payload string
        """
        inbox = self._build_inbox()
        s = self.subscribe(inbox, callback)
        self.unsubscribe(s, 1)
        self.publish(subject, msg, inbox)

        return s