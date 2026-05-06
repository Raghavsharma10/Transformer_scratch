def get_authorizations_by_search(self, authorization_query, authorization_search):
        """Pass through to provider AuthorizationSearchSession.get_authorizations_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_authorizations_by_search(authorization_query, authorization_search)