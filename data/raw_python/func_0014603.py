def _load_item(self, key):
        '''Load the specified item from the [flask] section. Type is
        determined by the type of the equivalent value in app.default_config
        or string if unknown.'''
        key_u   = key.upper()
        default = current_app.default_config.get(key_u)

        # One of the default config vars is a timedelta - interpret it
        # as an int and construct using it
        if isinstance(default, datetime.timedelta):
            current_app.config[key_u] = datetime.timedelta(self.getint('flask', key))
        elif isinstance(default, bool):
            current_app.config[key_u] = self.getboolean('flask', key)
        elif isinstance(default, float):
            current_app.config[key_u] = self.getfloat('flask', key)
        elif isinstance(default, int):
            current_app.config[key_u] = self.getint('flask', key)
        else:
            # All the string keys need to be coerced into str()
            # because Flask expects some of them not to be unicode
            current_app.config[key_u] = str(self.get('flask', key))