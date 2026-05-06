def validate(self, value):
        """
        From a value available on the remote server, the method returns the
        complete item matching the value.
        If case the value is not available on the server side or filtered
        through :meth:`item`, the class:`agnocomplete.exceptions.ItemNotFound`
        is raised.
        """

        url = self.get_item_url(value)
        try:
            data = self.http_call(url=url)
        except requests.HTTPError:
            raise ItemNotFound()

        data = self.get_http_result(data)

        try:
            self.item(data)
        except SkipItem:
            raise ItemNotFound()

        return value