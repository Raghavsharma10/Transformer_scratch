def get_grade_entries_by_search(self, grade_entry_query, grade_entry_search):
        """Pass through to provider GradeEntrySearchSession.get_grade_entries_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_grade_entries_by_search(grade_entry_query, grade_entry_search)