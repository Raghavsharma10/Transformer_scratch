def get_functions_by_search(self, function_query, function_search):
        """Pass through to provider FunctionSearchSession.get_functions_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_functions_by_search(function_query, function_search)