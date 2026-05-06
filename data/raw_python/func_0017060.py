def set(self, subscribed, ignored):
        """Set the user's subscription for this subscription

        :param bool subscribed: (required), determines if notifications should
            be received from this thread.
        :param bool ignored: (required), determines if notifications should be
            ignored from this thread.
        """
        sub = {'subscribed': subscribed, 'ignored': ignored}
        json = self._json(self._put(self._api, data=dumps(sub)), 200)
        self.__init__(json, self._session)