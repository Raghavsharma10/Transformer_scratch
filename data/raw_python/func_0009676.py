def poll(self, timeout=30, interval=2):
        """Return the results when they become ready.

        :param int timeout: the maximum time to wait for the results
        :param float interval: the number of seconds between checks
        :return: the membership request result
        :rtype: :class:`~groupy.api.memberships.MembershipResult.Results`
        """
        time.sleep(interval)
        start = time.time()
        while time.time() - start < timeout and not self.is_ready():
            time.sleep(interval)
        return self.get()