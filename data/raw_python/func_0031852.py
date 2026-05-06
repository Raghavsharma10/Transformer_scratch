def get_waittime(self):
        """Return the appropriate time to wait, if we sent too many messages

        :returns: the time to wait in seconds
        :rtype: :class:`float`
        :raises: None
        """
        now = time.time()
        self.sentmessages.appendleft(now)
        if len(self.sentmessages) == self.sentmessages.maxlen:
            # check if the oldes message is older than
            # limited by self.limitinterval
            oldest = self.sentmessages[-1]
            waittime = self.limitinterval - (now - oldest)
            if waittime > 0:
                return waittime + 1  # add a little buffer
        return 0