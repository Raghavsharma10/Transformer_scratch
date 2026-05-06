def get_subscribers(self, 
            order="created_at desc",
            offset=None,
            count=None):
        """Returns a list of subscribers.
        
        List is sorted by most-recent-to-subsribe, starting at an optional integer ``offset``, and optionally limited to the first ``count`` items (in sorted order).

        Returned data includes various statistics about each subscriber, e.g., ``total_sent``, ``total_opens``, ``total_clicks``.
        """
        req_data = [ None, order, fmt_paging(offset, count)]
        return self.request("query:Contact.stats", req_data)