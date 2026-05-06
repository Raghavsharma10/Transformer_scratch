def retrieveVals(self):
        """Retrieve values for graphs."""
        opcinfo = OPCinfo(self._host, self._port, self._user, self._password, 
                          self._monpath, self._ssl)
        stats = opcinfo.getAllStats()
        
        if self.hasGraph('php_opc_memory') and stats:
            mem = stats['memory_usage']
            keys = ('used_memory', 'wasted_memory', 'free_memory')
            map(lambda k:self.setGraphVal('php_opc_memory',k,mem[k]), keys)

        if self.hasGraph('php_opc_opcache_statistics') and stats:
            st = stats['opcache_statistics']
            self.setGraphVal('php_opc_opcache_statistics', 'hits', 
                             st['hits'])
            self.setGraphVal('php_opc_opcache_statistics', 'misses', 
                             st['misses'])

        if self.hasGraph('php_opc_opcache_hitrate') and stats:
            st = stats['opcache_statistics']
            self.setGraphVal('php_opc_opcache_hitrate', 'opcache_hit_rate',
                             st['opcache_hit_rate'])

        if self.hasGraph('php_opc_key_status') and stats:
            st = stats['opcache_statistics']
            wasted = st['num_cached_keys'] - st['num_cached_scripts']
            free = st['max_cached_keys'] - st['num_cached_keys']
            self.setGraphVal('php_opc_key_status', 'num_cached_scripts', st['num_cached_scripts'])
            self.setGraphVal('php_opc_key_status', 'num_wasted_keys', wasted)
            self.setGraphVal('php_opc_key_status', 'num_free_keys', free)