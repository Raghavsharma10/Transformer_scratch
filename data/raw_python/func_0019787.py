def envGet(self, name, default=None, conv=None):
        """Return value for environment variable or None.  
        
        @param name:    Name of environment variable.
        @param default: Default value if variable is undefined.
        @param conv:    Function for converting value to desired type.
        @return:        Value of environment variable.
        
        """
        if self._env.has_key(name):
            if conv is not None:
                return conv(self._env.get(name))
            else:
                return self._env.get(name)
        else:
            return default