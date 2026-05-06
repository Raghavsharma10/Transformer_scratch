def get_failed_requests(self, results):
        """Return the requests that failed.

        :param results: the results of a membership request check
        :type results: :class:`list`
        :return: the failed requests
        :rtype: generator
        """
        data = {member['guid']: member for member in results}
        for request in self.requests:
            if request['guid'] not in data:
                yield request