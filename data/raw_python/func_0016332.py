def send(self, **extra):
        """
        Sends email batch.

        :return: Information about sent emails.
        :rtype: `list`
        """
        emails = self.as_dict(**extra)
        responses = [self._manager._send_batch(*batch) for batch in chunks(emails, self.MAX_SIZE)]
        return sum(responses, [])