def send_notification(self, subject="", message="", sender="", source=None, actions=None):
        """
        Sends a notification. Blocks as long as necessary.

        :param subject: The subject.
        :type subject: str
        :param message: The message.
        :type message: str
        :param sender: The sender.
        :type sender: str
        :param source: The source of the notification
        :type source: .LegacyNotification.Source
        :param actions Actions to be sent with a notification (list of TimelineAction objects)
        :type actions list
        """
        if self._pebble.firmware_version.major < 3:
            self._send_legacy_notification(subject, message, sender, source)
        else:
            self._send_modern_notification(subject, message, sender, source, actions)