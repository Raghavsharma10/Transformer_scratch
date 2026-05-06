def get_objectives_by_search(self, objective_query, objective_search):
        """Pass through to provider ObjectiveSearchSession.get_objectives_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_objectives_by_search(objective_query, objective_search)