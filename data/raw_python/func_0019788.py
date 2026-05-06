def envGetList(self, name, attr_regex = '^\w+$', conv=None):
        """Parse the plugin environment variables to return list from variable
        with name list_<name>. The value of the variable must be a comma 
        separated list of items.
        
        @param name:       Name of list.
                           (Also determines suffix for environment variable name.)
        @param attr_regex: If the regex is defined, the items in the list are 
                           ignored unless they comply with the format dictated 
                           by the match regex.
        @param conv:       Function for converting value to desired type.
        @return:           List of items.
        
        """
        key = "list_%s" % name
        item_list = []
        if self._env.has_key(key):
            if attr_regex:
                recomp = re.compile(attr_regex)
            else:
                recomp = None
            for attr in self._env[key].split(','):
                attr = attr.strip()
                if recomp is None or recomp.search(attr):
                    if conv is not None:
                        item_list.append(conv(attr))
                    else:
                        item_list.append(attr)
                    
        return item_list