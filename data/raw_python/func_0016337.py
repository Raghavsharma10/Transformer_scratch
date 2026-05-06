def all(
        self,
        count=500,
        offset=0,
        type=None,
        inactive=None,
        emailFilter=None,
        tag=None,
        messageID=None,
        fromdate=None,
        todate=None,
    ):
        """
        Returns many bounces.

        :param int count: Number of bounces to return per request.
        :param int offset: Number of bounces to skip.
        :param str type: Filter by type of bounce.
        :param bool inactive: Filter by emails that were deactivated by Postmark due to the bounce.
        :param str emailFilter: Filter by email address.
        :param str tag: Filter by tag.
        :param str messageID: Filter by messageID.
        :param date fromdate: Filter messages starting from the date specified (inclusive).
        :param date todate: Filter messages up to the date specified (inclusive).
        :return: A list of :py:class:`Bounce` instances.
        :rtype: `list`
        """
        responses = self.call_many(
            "GET",
            "/bounces/",
            count=count,
            offset=offset,
            type=type,
            inactive=inactive,
            emailFilter=emailFilter,
            tag=tag,
            messageID=messageID,
            fromdate=fromdate,
            todate=todate,
        )
        return self.expand_responses(responses, "Bounces")