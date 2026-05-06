def _get_context_key(self, **kwargs):
        """ Get value of `self._resource.parent.id_name` from :kwargs: """
        return str(kwargs.get(self._resource.parent.id_name))