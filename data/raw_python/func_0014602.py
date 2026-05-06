def parse_flask_section(self):
        '''Parse the [flask] section of your config and hand off the config
        to the app in context.

        Config vars should have the same name as their flask equivalent except
        in all lower-case.'''
        if self.has_section('flask'):
            for item in self.items('flask'):
                self._load_item(item[0])
        else:
            warnings.warn("No [flask] section found in config")