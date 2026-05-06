def envCheckFlag(self, name, default = False):
        """Check graph flag for enabling / disabling attributes through
        the use of <name> environment variable.
        
        @param name:    Name of flag.
                        (Also determines the environment variable name.)
        @param default: Boolean (True or False). Default value for flag.
        @return:        Return True if the flag is enabled.
        
        """
        if self._flags.has_key(name):
            return self._flags[name]
        else:
            val = self._env.get(name)
            if val is None:
                return default
            elif val.lower() in ['yes', 'on']:
                self._flags[name] = True
                return True
            elif val.lower() in ['no', 'off']:
                self._flags[name] = False
                return False
            else:
                raise AttributeError("Value for flag %s, must be yes, no, on or off" 
                                     % name)