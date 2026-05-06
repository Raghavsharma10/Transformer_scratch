def to_json(self):
        """Short cut for JSON response service data.

        Returns:
            Dict that implements JSON interface.
        """

        web_resp = collections.OrderedDict()

        web_resp['status_code'] = self.status_code
        web_resp['status_text'] = dict(HTTP_CODES).get(self.status_code)
        web_resp['data'] = self.data if self.data is not None else {}
        web_resp['errors'] = self.errors or []

        return web_resp