def get_grade_systems_by_search(self, grade_system_query, grade_system_search):
        """Pass through to provider GradeSystemSearchSession.get_grade_systems_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_grade_systems_by_search(grade_system_query, grade_system_search)