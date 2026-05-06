def collection_options(self, **kwargs):
        """ Handle collection item OPTIONS request. """
        methods = self._get_handled_methods(self._collection_actions)
        return self._set_options_headers(methods)