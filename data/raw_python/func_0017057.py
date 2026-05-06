def delete_subscription(self):
        """Delete subscription for this thread.

        :returns: bool
        """
        url = self._build_url('subscription', base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)