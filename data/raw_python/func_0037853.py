def create_global_notification(self, account_id, account_notification_end_at, account_notification_subject, account_notification_message, account_notification_start_at, account_notification_icon=None, account_notification_roles=None):
        """
        Create a global notification.

        Create and return a new global notification for an account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - account_notification[subject]
        """The subject of the notification."""
        data["account_notification[subject]"] = account_notification_subject

        # REQUIRED - account_notification[message]
        """The message body of the notification."""
        data["account_notification[message]"] = account_notification_message

        # REQUIRED - account_notification[start_at]
        """The start date and time of the notification in ISO8601 format.
        e.g. 2014-01-01T01:00Z"""
        data["account_notification[start_at]"] = account_notification_start_at

        # REQUIRED - account_notification[end_at]
        """The end date and time of the notification in ISO8601 format.
        e.g. 2014-01-01T01:00Z"""
        data["account_notification[end_at]"] = account_notification_end_at

        # OPTIONAL - account_notification[icon]
        """The icon to display with the notification.
        Note: Defaults to warning."""
        if account_notification_icon is not None:
            self._validate_enum(account_notification_icon, ["warning", "information", "question", "error", "calendar"])
            data["account_notification[icon]"] = account_notification_icon

        # OPTIONAL - account_notification_roles
        """The role(s) to send global notification to.  Note:  ommitting this field will send to everyone
        Example:
          account_notification_roles: ["StudentEnrollment", "TeacherEnrollment"]"""
        if account_notification_roles is not None:
            data["account_notification_roles"] = account_notification_roles

        self.logger.debug("POST /api/v1/accounts/{account_id}/account_notifications with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/account_notifications".format(**path), data=data, params=params, no_data=True)