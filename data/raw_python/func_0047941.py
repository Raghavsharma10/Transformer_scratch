def get_resource_relationships_by_search(self, resource_relationship_query, resource_relationship_search):
        """Pass through to provider ResourceRelationshipSearchSession.get_resource_relationships_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_resource_relationships_by_search(resource_relationship_query, resource_relationship_search)