def url(self):
        """
        We will always check if this song file exists in local library,
        if true, we return the url of the local file.

        .. note::

            As netease song url will be expired after a period of time,
            we can not use static url here. Currently, we assume that the
            expiration time is 20 minutes, after the url expires, it
            will be automaticly refreshed.
        """
        local_path = self._find_in_local()
        if local_path:
            return local_path

        if not self._url:
            self._refresh_url()
        elif time.time() > self._expired_at:
            logger.info('song({}) url is expired, refresh...'.format(self))
            self._refresh_url()
        return self._url