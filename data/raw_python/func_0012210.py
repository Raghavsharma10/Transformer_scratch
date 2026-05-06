def get_notification(self, notification_id):
        """
        returns a specific notification by given id

        :param str notification_id: the ID of the notification
        """
        log.debug("getting notification '{}'...".format(notification_id))
        cmd, url = DEVICE_URLS["get_notification"]
        return self._exec(cmd, url.replace(":id", notification_id))