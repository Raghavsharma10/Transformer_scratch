def envRegisterFilter(self, name, attr_regex = '^\w+$', default = True):
        """Register filter for including, excluding attributes in graphs through 
        the use of include_<name> and exclude_<name> environment variables.
        The value of the variables must be a comma separated list of items. 
        
        @param name:       Name of filter.
                           (Also determines suffix for environment variable name.)
        @param attr_regex: Regular expression string for checking valid items.
        @param default:    Filter default. Applies when the include list is not 
                           defined and the attribute is not in the exclude list.
        
        """
        attrs = {}
        for prefix in ('include', 'exclude'):
            key = "%s_%s" % (prefix, name)
            val = self._env.get(key)
            if val:
                attrs[prefix] = [attr.strip() for attr in val.split(',')]
            else:
                attrs[prefix] = []
        self._filters[name] = MuninAttrFilter(attrs['include'], attrs['exclude'], 
                                              attr_regex, default)