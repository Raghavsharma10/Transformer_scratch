def sync(self):
        """ Send and fetch all outstanding messages.

        :return: 2-tuple of number of detail messages and number of summary messages fetched
        """
        self.send()
        detail_count = summary_count = 0
        while self.responses:
            response = self.responses[0]
            while not response.complete:
                detail_delta, summary_delta = self.fetch()
                detail_count += detail_delta
                summary_count += summary_delta
        return detail_count, summary_count