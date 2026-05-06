def get_qualifiers_by_search(self, qualifier_query, qualifier_search):
        """Pass through to provider QualifierSearchSession.get_qualifiers_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_qualifiers_by_search(qualifier_query, qualifier_search)