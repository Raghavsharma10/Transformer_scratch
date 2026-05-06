def list_migration_issues_groups(self, group_id, content_migration_id):
        """
        List migration issues.

        Returns paginated migration issues
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - content_migration_id
        """ID"""
        path["content_migration_id"] = content_migration_id

        self.logger.debug("GET /api/v1/groups/{group_id}/content_migrations/{content_migration_id}/migration_issues with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/groups/{group_id}/content_migrations/{content_migration_id}/migration_issues".format(**path), data=data, params=params, all_pages=True)