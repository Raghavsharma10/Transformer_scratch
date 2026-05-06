def _update(self, **kwargs):
        """Update some attributes.

        If a 'settings' attribute is passed as a dict, then it updates the
        content of the settings, if any, instead of completely overwriting it.
        """
        for key, value in _iteritems(kwargs):
            if key == 'settings':
                if isinstance(value, dict):
                    if self.settings is None:
                        self.settings = Settings(**value)
                    else:
                        self.settings._update(**value)
                else:
                    self.settings = copy.deepcopy(value)
            else:
                setattr(self, key, value)