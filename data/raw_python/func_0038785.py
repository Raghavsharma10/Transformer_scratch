def list_migration_issues_users(self, user_id, content_migration_id):
        """
        List migration issues.

        Returns paginated migration issues
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - content_migration_id
        """ID"""
        path["content_migration_id"] = content_migration_id

        self.logger.debug("GET /api/v1/users/{user_id}/content_migrations/{content_migration_id}/migration_issues with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/content_migrations/{content_migration_id}/migration_issues".format(**path), data=data, params=params, all_pages=True)