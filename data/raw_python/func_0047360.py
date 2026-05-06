def use_plenary_item_view(self):
        """Pass through to provider ItemLookupSession.use_plenary_item_view"""
        self._object_views['item'] = PLENARY
        # self._get_provider_session('item_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_item_view()
            except AttributeError:
                pass