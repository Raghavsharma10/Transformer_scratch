def get_received(self, age=None, for_all=True):
        """Retrieve a list of transfers sent to you or your company
         from other people.

        :param age: between 1 and 90 days.
        :param for_all: If ``True`` will return received files for
         all users in the same business. (Available for business account
         members only).
        :type age: ``int``
        :type for_all: ``bool``
        :rtype: ``list`` of :class:`Transfer` objects.
        """

        method, url = get_URL('received_get')

        if age:
            if not isinstance(age, int) or age < 0 or age > 90:
                raise FMBaseError('Age must be <int> between 0-90')

            past = datetime.utcnow() - timedelta(days=age)
            age = timegm(past.utctimetuple())

        payload = {
            'apikey': self.config.get('apikey'),
            'logintoken': self.session.cookies.get('logintoken'),
            'getForAllUsers': for_all,
            'from': age
            }

        res = getattr(self.session, method)(url, params=payload)

        if res.status_code == 200:
            return self._restore_transfers(res)

        hellraiser(res)