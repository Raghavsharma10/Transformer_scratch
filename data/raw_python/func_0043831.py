def get_urls(self, order="total_clicks desc", offset=None, count=None):
        """Returns a list of URLs you've included in messages.

        List is sorted by ``total_clicks``, starting at an optional integer ``offset``, and optionally limited to the first ``count`` items.
        """
        req_data = [ None, order, fmt_paging(offset, count) ]
        return self.request("query:Message_Url", req_data)