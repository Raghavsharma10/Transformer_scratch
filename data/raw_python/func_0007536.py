def has_expired(self, lifetime, now=None):
        """Report if the session key has expired.

        :param lifetime: A :class:`datetime.timedelta` that specifies the
                         maximum age this :class:`SessionID` should be checked
                         against.
        :param now: If specified, use this :class:`~datetime.datetime` instance
                         instead of :meth:`~datetime.datetime.utcnow()` as the
                         current time.
        """
        now = now or datetime.utcnow()
        return now > self.created + lifetime