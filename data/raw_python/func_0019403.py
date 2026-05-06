def unsubscribe(self, subscription, max=None):
        """
        Unsubscribe will remove interest in the given subject. If max is
        provided an automatic Unsubscribe that is processed by the server
        when max messages have been received

        Args:
            subscription (pynats.Subscription): a Subscription object
            max (int=None): number of messages
        """
        if max is None:
            self._send('UNSUB %d' % subscription.sid)
            self._subscriptions.pop(subscription.sid)
        else:
            subscription.max = max
            self._send('UNSUB %d %s' % (subscription.sid, max))