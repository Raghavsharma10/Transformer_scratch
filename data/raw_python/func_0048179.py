def use_plenary_composition_view(self):
        """Pass through to provider CompositionLookupSession.use_plenary_composition_view"""
        self._object_views['composition'] = PLENARY
        # self._get_provider_session('composition_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_composition_view()
            except AttributeError:
                pass