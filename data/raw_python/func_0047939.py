def get_resources_by_search(self, resource_query, resource_search):
        """Pass through to provider ResourceSearchSession.get_resources_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_resources_by_search(resource_query, resource_search)