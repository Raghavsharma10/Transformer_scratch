def subscription(self):
        """Checks the status of the user's subscription to this thread.

        :returns: :class:`Subscription <Subscription>`
        """
        url = self._build_url('subscription', base_url=self._api)
        json = self._json(self._get(url), 200)
        return Subscription(json, self) if json else None