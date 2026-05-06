def get_messages(self, new=True):
        """
        Return all messages sent to this user, formatted as a notification.

        :param new: False for all notifications, True for only non-viewed
            notifications.
        """
        url = (self._imgur._base_url + "/3/account/{0}/notifications/"
               "messages".format(self.name))
        result = self._imgur._send_request(url, params=locals(),
                                           needs_auth=True)
        return [Notification(msg_dict, self._imgur, has_fetched=True) for
                msg_dict in result]