def set_subscription(self, subscribed, ignored):
        """Set the user's subscription for this thread

        :param bool subscribed: (required), determines if notifications should
            be received from this thread.
        :param bool ignored: (required), determines if notifications should be
            ignored from this thread.
        :returns: :class:`Subscription <Subscription>`
        """
        url = self._build_url('subscription', base_url=self._api)
        sub = {'subscribed': subscribed, 'ignored': ignored}
        json = self._json(self._put(url, data=dumps(sub)), 200)
        return Subscription(json, self) if json else None