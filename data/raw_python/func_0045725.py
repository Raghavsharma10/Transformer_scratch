def get_log_entries_by_search(self, log_entry_query, log_entry_search):
        """Pass through to provider LogEntrySearchSession.get_log_entries_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_log_entries_by_search(log_entry_query, log_entry_search)