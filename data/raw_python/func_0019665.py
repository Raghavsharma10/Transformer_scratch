def retrieveVals(self):
        """Retrieve values for graphs."""
        apcinfo = APCinfo(self._host, self._port, self._user, self._password, 
                          self._monpath, self._ssl, self._extras)
        stats = apcinfo.getAllStats()
        
        if self.hasGraph('php_apc_memory') and stats:
            filecache = stats['cache_sys']['mem_size']
            usercache = stats['cache_user']['mem_size']
            total = stats['memory']['seg_size'] * stats['memory']['num_seg']
            free = stats['memory']['avail_mem']
            other = total - free - filecache - usercache 
            self.setGraphVal('php_apc_memory', 'filecache', filecache)
            self.setGraphVal('php_apc_memory', 'usercache', usercache)
            self.setGraphVal('php_apc_memory', 'other', other)
            self.setGraphVal('php_apc_memory', 'free', free)
        if self.hasGraph('php_apc_items') and stats:
            self.setGraphVal('php_apc_items', 'filecache', 
                             stats['cache_sys']['num_entries'])
            self.setGraphVal('php_apc_items', 'usercache', 
                             stats['cache_user']['num_entries'])
        if self.hasGraph('php_apc_reqs_filecache') and stats:
            self.setGraphVal('php_apc_reqs_filecache', 'hits', 
                             stats['cache_sys']['num_hits'])
            self.setGraphVal('php_apc_reqs_filecache', 'misses', 
                             stats['cache_sys']['num_misses'])
            self.setGraphVal('php_apc_reqs_filecache', 'inserts', 
                             stats['cache_sys']['num_inserts'])
        if self.hasGraph('php_apc_reqs_usercache') and stats:
            self.setGraphVal('php_apc_reqs_usercache', 'hits', 
                             stats['cache_user']['num_hits'])
            self.setGraphVal('php_apc_reqs_usercache', 'misses', 
                             stats['cache_user']['num_misses'])
            self.setGraphVal('php_apc_reqs_usercache', 'inserts', 
                             stats['cache_user']['num_inserts'])
        if self.hasGraph('php_apc_expunge') and stats:
            self.setGraphVal('php_apc_expunge', 'filecache', 
                             stats['cache_sys']['expunges'])
            self.setGraphVal('php_apc_expunge', 'usercache', 
                             stats['cache_user']['expunges'])
        if self.hasGraph('php_apc_mem_util_frag'):
            self.setGraphVal('php_apc_mem_util_frag', 'util', 
                             stats['memory']['utilization_ratio'] * 100)
            self.setGraphVal('php_apc_mem_util_frag', 'frag', 
                             stats['memory']['fragmentation_ratio'] * 100)
        if self.hasGraph('php_apc_mem_frag_count'):
            self.setGraphVal('php_apc_mem_frag_count', 'num',
                             stats['memory']['fragment_count'])
        if self.hasGraph('php_apc_mem_frag_avgsize'):
            self.setGraphVal('php_apc_mem_frag_avgsize', 'size',
                             stats['memory']['fragment_avg_size'])