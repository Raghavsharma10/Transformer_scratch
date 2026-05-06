def update_migration_issue_users(self, id, user_id, workflow_state, content_migration_id):
        """
        Update a migration issue.

        Update the workflow_state of a migration issue
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

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - workflow_state
        """Set the workflow_state of the issue."""
        self._validate_enum(workflow_state, ["active", "resolved"])
        data["workflow_state"] = workflow_state

        self.logger.debug("PUT /api/v1/users/{user_id}/content_migrations/{content_migration_id}/migration_issues/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/{user_id}/content_migrations/{content_migration_id}/migration_issues/{id}".format(**path), data=data, params=params, single_item=True)