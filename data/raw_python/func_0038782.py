def list_migration_issues_accounts(self, account_id, content_migration_id):
        """
        List migration issues.

        Returns paginated migration issues
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - content_migration_id
        """ID"""
        path["content_migration_id"] = content_migration_id

        self.logger.debug("GET /api/v1/accounts/{account_id}/content_migrations/{content_migration_id}/migration_issues with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/content_migrations/{content_migration_id}/migration_issues".format(**path), data=data, params=params, all_pages=True)