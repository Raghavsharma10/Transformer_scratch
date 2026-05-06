def enabled(self):
        """Returns True if this notification is enabled based on the value
        of Notification model instance.

        Note: Notification names/display_names are persisted in the
        "Notification" model where each mode instance can be flagged
        as enabled or not, and are selected/subscribed to by
        each user in their user profile.

        See also: `site_notifications.update_notification_list`
        """
        if not self._notification_enabled:
            self._notification_enabled = self.notification_model.enabled
        return self._notification_enabled