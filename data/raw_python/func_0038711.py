def update_account(self, id, account_default_group_storage_quota_mb=None, account_default_storage_quota_mb=None, account_default_time_zone=None, account_default_user_storage_quota_mb=None, account_name=None, account_services=None, account_settings_lock_all_announcements_locked=None, account_settings_lock_all_announcements_value=None, account_settings_restrict_student_future_listing_locked=None, account_settings_restrict_student_future_listing_value=None, account_settings_restrict_student_future_view_locked=None, account_settings_restrict_student_future_view_value=None, account_settings_restrict_student_past_view_locked=None, account_settings_restrict_student_past_view_value=None):
        """
        Update an account.

        Update an existing account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - account[name]
        """Updates the account name"""
        if account_name is not None:
            data["account[name]"] = account_name

        # OPTIONAL - account[default_time_zone]
        """The default time zone of the account. Allowed time zones are
        {http://www.iana.org/time-zones IANA time zones} or friendlier
        {http://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html Ruby on Rails time zones}."""
        if account_default_time_zone is not None:
            data["account[default_time_zone]"] = account_default_time_zone

        # OPTIONAL - account[default_storage_quota_mb]
        """The default course storage quota to be used, if not otherwise specified."""
        if account_default_storage_quota_mb is not None:
            data["account[default_storage_quota_mb]"] = account_default_storage_quota_mb

        # OPTIONAL - account[default_user_storage_quota_mb]
        """The default user storage quota to be used, if not otherwise specified."""
        if account_default_user_storage_quota_mb is not None:
            data["account[default_user_storage_quota_mb]"] = account_default_user_storage_quota_mb

        # OPTIONAL - account[default_group_storage_quota_mb]
        """The default group storage quota to be used, if not otherwise specified."""
        if account_default_group_storage_quota_mb is not None:
            data["account[default_group_storage_quota_mb]"] = account_default_group_storage_quota_mb

        # OPTIONAL - account[settings][restrict_student_past_view][value]
        """Restrict students from viewing courses after end date"""
        if account_settings_restrict_student_past_view_value is not None:
            data["account[settings][restrict_student_past_view][value]"] = account_settings_restrict_student_past_view_value

        # OPTIONAL - account[settings][restrict_student_past_view][locked]
        """Lock this setting for sub-accounts and courses"""
        if account_settings_restrict_student_past_view_locked is not None:
            data["account[settings][restrict_student_past_view][locked]"] = account_settings_restrict_student_past_view_locked

        # OPTIONAL - account[settings][restrict_student_future_view][value]
        """Restrict students from viewing courses before start date"""
        if account_settings_restrict_student_future_view_value is not None:
            data["account[settings][restrict_student_future_view][value]"] = account_settings_restrict_student_future_view_value

        # OPTIONAL - account[settings][restrict_student_future_view][locked]
        """Lock this setting for sub-accounts and courses"""
        if account_settings_restrict_student_future_view_locked is not None:
            data["account[settings][restrict_student_future_view][locked]"] = account_settings_restrict_student_future_view_locked

        # OPTIONAL - account[settings][lock_all_announcements][value]
        """Disable comments on announcements"""
        if account_settings_lock_all_announcements_value is not None:
            data["account[settings][lock_all_announcements][value]"] = account_settings_lock_all_announcements_value

        # OPTIONAL - account[settings][lock_all_announcements][locked]
        """Lock this setting for sub-accounts and courses"""
        if account_settings_lock_all_announcements_locked is not None:
            data["account[settings][lock_all_announcements][locked]"] = account_settings_lock_all_announcements_locked

        # OPTIONAL - account[settings][restrict_student_future_listing][value]
        """Restrict students from viewing future enrollments in course list"""
        if account_settings_restrict_student_future_listing_value is not None:
            data["account[settings][restrict_student_future_listing][value]"] = account_settings_restrict_student_future_listing_value

        # OPTIONAL - account[settings][restrict_student_future_listing][locked]
        """Lock this setting for sub-accounts and courses"""
        if account_settings_restrict_student_future_listing_locked is not None:
            data["account[settings][restrict_student_future_listing][locked]"] = account_settings_restrict_student_future_listing_locked

        # OPTIONAL - account[services]
        """Give this a set of keys and boolean values to enable or disable services matching the keys"""
        if account_services is not None:
            data["account[services]"] = account_services

        self.logger.debug("PUT /api/v1/accounts/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/accounts/{id}".format(**path), data=data, params=params, single_item=True)