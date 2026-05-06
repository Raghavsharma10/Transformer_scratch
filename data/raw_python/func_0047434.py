def get_gradebook_columns_by_search(self, gradebook_column_query, gradebook_column_search):
        """Pass through to provider GradebookColumnSearchSession.get_gradebook_columns_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_gradebook_columns_by_search(gradebook_column_query, gradebook_column_search)