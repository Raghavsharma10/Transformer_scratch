def get_notification(self, id):
        """
        Return a Notification object.

        :param id: The id of the notification object to return.
        """
        url = self._base_url + "/3/notification/{0}".format(id)
        resp = self._send_request(url)
        return Notification(resp, self)