def as_dict(self):
        """
        Return the contents as dictionary, for client-side export.
        The dictionary contains the fields:

        * ``slot``
        * ``title``
        * ``role``
        * ``fallback_language``
        * ``allowed_plugins``
        """
        plugins = self.get_allowed_plugins()
        return {
            'slot': self.slot,
            'title': self.title,
            'role': self.role,
            'fallback_language': self.fallback_language,
            'allowed_plugins': [plugin.name for plugin in plugins],
        }