def use_plenary_activity_view(self):
        """Pass through to provider ActivityLookupSession.use_plenary_activity_view"""
        self._object_views['activity'] = PLENARY
        # self._get_provider_session('activity_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_activity_view()
            except AttributeError:
                pass