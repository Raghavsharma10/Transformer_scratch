def _get_setting(self, setting, default=None, name=None, inherit=True):
        """ Helper function to fetch settings, inheriting from the base """
        if name is None:
            name = self.name
        if name == 'DEFAULT':
            return self._settings.get('webpack.{0}'.format(setting), default)
        else:
            val = self._settings.get('webpack.{0}.{1}'.format(name, setting),
                                     SENTINEL)
            if val is SENTINEL:
                if inherit:
                    return self._get_setting(setting, default, 'DEFAULT')
                else:
                    return default
            else:
                return val