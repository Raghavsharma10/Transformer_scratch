def list_all(self, before_id=None, since_id=None, **kwargs):
        """Return all direct messages.

        The messages come in reversed order (newest first). Note you can only
        provide _one_ of ``before_id``, ``since_id``.

        :param str before_id: message ID for paging backwards
        :param str since_id: message ID for most recent messages since
        :return: direct messages
        :rtype: generator
        """
        return self.list(before_id=before_id, since_id=since_id, **kwargs).autopage()