def get_replies(self, new=True):
        """
        Return all reply notifications for this user.

        :param new: False for all notifications, True for only non-viewed
            notifications.
        """
        url = (self._imgur._base_url + "/3/account/{0}/"
               "notifications/replies".format(self.name))
        return self._imgur._send_request(url, needs_auth=True)