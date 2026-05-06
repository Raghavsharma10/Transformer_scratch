def get_messages(self,
            statuses=DEFAULT_MESSAGE_STATUSES,
            order="sent_at desc",
            offset=None,
            count=None,
            content=False):
        """Returns a list of messages your account sent.
        
        Messages are sorted by ``order``, starting at an optional integer ``offset``, and optionally limited to the first ``count`` items (in sorted order).

        Returned data includes various statistics about each message, e.g., ``total_opens``, ``open_rate``, ``total_clicks``, ``unsubs``, ``soft_bounces``. If ``content=True``, the returned data will also include HTML content of each message.
        """

        req_data = [ { "status": statuses }, order, fmt_paging(offset, count) ]
        service = "query:Message.stats"
        if content: service += ", Message.content"
        return self.request(service, req_data)