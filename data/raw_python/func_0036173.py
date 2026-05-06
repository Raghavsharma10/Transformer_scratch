def get_list(self, section, option):
        """
        Convert string value to list object.
        """
        if self.has_option(section, option):
            return self.get(section, option).replace(' ', '').split(',')
        else:
            raise KeyError('{} with {} does not exist.'.format(section,
                                                               option))