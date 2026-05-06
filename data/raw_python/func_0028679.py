def events(self, *etypes, idle_timeout=None):
        """
        returns a blocking generator yielding Slack event objects
        params:
         - etypes(str): If defined, Slack event type(s) not matching
           the filter will be ignored. See https://api.slack.com/events for
           a listing of valid event types.
         - idle_timeout(int): optional maximum amount of time (in seconds)
           to wait between events before returning
        """

        while self._state != STATE_STOPPED:
            try:
                yield self.get_event(*etypes, timeout=idle_timeout)
            except Queue.Empty:
                log.info('idle timeout reached for events()')
                return