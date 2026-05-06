def _get_hierarchy_session(self, hierarchy_id):
        """Returns a hierarchy traversal session for the hierarchy"""
        hierarchy_mgr = self._get_provider_manager('HIERARCHY', local=True)
        return hierarchy_mgr.get_hierarchy_traversal_session_for_hierarchy(
            hierarchy_id,
            proxy=self._proxy)