def get_event(self, *etypes, timeout=None):
        """
        Return a single event object or block until an event is
        received and return it.
         - etypes(str): If defined, Slack event type(s) not matching
           the filter will be ignored. See https://api.slack.com/events for
           a listing of valid event types.
         - timeout(int): Max time, in seconds, to block waiting for new event
        """
        self._validate_etypes(*etypes)
        start = time.time()
        e = self._eventq.get(timeout=timeout)

        if isinstance(e, Exception):
            raise e

        self._stats['events_recieved'] += 1
        if etypes and e.type not in etypes:
            if timeout:
                timeout -= time.time() - start
            log.debug('ignoring filtered event: {}'.format(e.json))
            self._stats['events_dropped'] += 1
            return self.get_event(*etypes, timeout=timeout)

        return e