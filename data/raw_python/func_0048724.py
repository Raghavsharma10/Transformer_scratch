def _can_for_object(self, func_name, object_id, method_name):
        """Checks if agent can perform function for object"""
        can_for_session = self._can(func_name)
        if (can_for_session or
                self._object_catalog_session is None or
                self._override_lookup_session is None):
            return can_for_session

        override_auths = self._override_lookup_session.get_authorizations_for_agent_and_function(
            self.get_effective_agent_id(),
            self._get_function_id(func_name))
        if not override_auths.available():
            return False

        if self._object_catalog_session is not None:
            catalog_ids = list(getattr(self._object_catalog_session, method_name)(object_id))
            for auth in override_auths:
                if auth.get_qualifier_id() in catalog_ids:
                    return True
        return False