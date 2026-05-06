def get_message_urls(self, message_id, order="total_clicks desc"):
        """Returns a list of URLs you've included in a specific message.

        List is sorted by ``total_clicks``, starting at an optional integer ``offset``, and optionally limited to the first ``count`` items.
        """
        req_data = [ { "message_id": str(message_id) }, order, None ]
        return self.request("query:Message_Url", req_data)