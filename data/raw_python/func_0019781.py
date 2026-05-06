def _parseEnv(self, env=None):
        """Private method for parsing through environment variables.
        
        Parses for environment variables common to all Munin Plugins:
            - MUNIN_STATEFILE
            - MUNIN_CAP_DIRTY_CONFIG
            - nested_graphs
        
        @param env: Dictionary of environment variables.
                    (Only used for testing. initialized automatically by 
                    constructor.
        
        """
        if not env:
            env = self._env
        if env.has_key('MUNIN_STATEFILE'):
            self._stateFile = env.get('MUNIN_STATEFILE')
        else:
            self._stateFile = '/tmp/munin-state-%s' % self.plugin_name
        if env.has_key('MUNIN_CAP_DIRTY_CONFIG'):
            self._dirtyConfig = True