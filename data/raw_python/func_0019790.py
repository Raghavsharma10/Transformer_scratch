def envCheckFilter(self, name, attr):
        """Check if a specific graph attribute is enabled or disabled through 
        the use of a filter based on include_<name> and exclude_<name> 
        environment variables.
        
        @param name: Name of the Filter.
        @param attr: Name of the Attribute.
        @return:     Return True if the attribute is enabled.
        
        """
        flt = self._filters.get(name)
        if flt:
            return flt.check(attr) 
        else:
            raise AttributeError("Undefined filter: %s" % name)