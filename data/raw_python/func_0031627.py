def api_version(self):
        """Cached version of :py:func:`~xmpp_backends.base.XmppBackendBase.get_api_version`."""

        now = datetime.utcnow()

        if self.version_cache_timestamp and self.version_cache_timestamp + self.version_cache_timeout > now:
            return self.version_cache_value  # we have a cached value

        self.version_cache_value = self.get_api_version()

        if self.minimum_version and self.version_cache_value < self.minimum_version:
            raise NotSupportedError('%s requires ejabberd >= %s' % (self.__class__.__name__,
                                                                    self.minimum_version))

        self.version_cache_timestamp = now
        return self.version_cache_value