def get(self):
        """Return the results now.

        :return: the membership request results
        :rtype: :class:`~groupy.api.memberships.MembershipResult.Results`
        :raises groupy.exceptions.ResultsNotReady: if the results are not ready
        :raises groupy.exceptions.ResultsExpired: if the results have expired
        """
        if self._expired_exception:
            raise self._expired_exception
        if self._not_ready_exception:
            raise self._not_ready_exception
        return self.results