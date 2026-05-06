def get_items_by_search(self, item_query, item_search):
        """Pass through to provider ItemSearchSession.get_items_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_items_by_search(item_query, item_search)