def create_new_sub_account(self, account_id, account_name, account_default_group_storage_quota_mb=None, account_default_storage_quota_mb=None, account_default_user_storage_quota_mb=None, account_sis_account_id=None):
        """
        Create a new sub-account.

        Add a new sub-account to a given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - account[name]
        """The name of the new sub-account."""
        data["account[name]"] = account_name

        # OPTIONAL - account[sis_account_id]
        """The account's identifier in the Student Information System."""
        if account_sis_account_id is not None:
            data["account[sis_account_id]"] = account_sis_account_id

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

        self.logger.debug("POST /api/v1/accounts/{account_id}/sub_accounts with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/sub_accounts".format(**path), data=data, params=params, single_item=True)