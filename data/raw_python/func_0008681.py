def refresh(self):
        """
        Refresh this objects attributes to the newest values.

        Attributes that weren't added to the object before, due to lazy
        loading, will be added by calling refresh.
        """
        resp = self._imgur._send_request(self._INFO_URL)
        self._populate(resp)
        self._has_fetched = True