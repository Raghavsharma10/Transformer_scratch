def get_proficiencies_by_search(self, proficiency_query, proficiency_search):
        """Pass through to provider ProficiencySearchSession.get_proficiencies_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_proficiencies_by_search(proficiency_query, proficiency_search)