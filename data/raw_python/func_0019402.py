def subscribe(self, subject, callback, queue=''):
        """
        Subscribe will express interest in the given subject. The subject can
        have wildcards (partial:*, full:>). Messages will be delivered to the
        associated callback.

        Args:
            subject (string): a string with the subject
            callback (function): callback to be called
        """
        s = Subscription(
            sid=self._next_sid,
            subject=subject,
            queue=queue,
            callback=callback,
            connetion=self
        )

        self._subscriptions[s.sid] = s
        self._send('SUB %s %s %d' % (s.subject, s.queue, s.sid))
        self._next_sid += 1

        return s