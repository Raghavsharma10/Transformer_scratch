def get_prefix(self):
        """ Each resource defined in config for pages as dict. This method
        returns key from config where located current resource.
        """
        for key, value in self.pages_config.items():
            if not hasattr(value, '__iter__'):
                value = (value, )
            for item in value:
                if type(self.node) == item\
                        or type(self.node) == getattr(item, 'model', None):
                    return key