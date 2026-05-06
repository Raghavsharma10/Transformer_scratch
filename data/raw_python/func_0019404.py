def publish(self, subject, msg, reply=None):
        """
        Publish publishes the data argument to the given subject.

        Args:
            subject (string): a string with the subject
            msg (string): payload string
            reply (string): subject used in the reply
        """
        if msg is None:
            msg = ''

        if reply is None:
            command = 'PUB %s %d' % (subject, len(msg))
        else:
            command = 'PUB %s %s %d' % (subject, reply, len(msg))

        self._send(command)
        self._send(msg)