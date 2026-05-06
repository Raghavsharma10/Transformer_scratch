def retrieveVals(self):
        """Retrieve values for graphs."""
        if self._stats is None:
            serverInfo = MemcachedInfo(self._host,  self._port, self._socket_file)
            stats = serverInfo.getStats()
        else:
            stats = self._stats
        if stats is None:
            raise Exception("Undetermined error accesing stats.")        
        stats['set_hits'] = stats.get('total_items')
        if stats.has_key('cmd_set') and stats.has_key('total_items'): 
            stats['set_misses'] = stats['cmd_set'] - stats['total_items']
        self.saveState(stats)
        if self.hasGraph('memcached_connections'):
            self.setGraphVal('memcached_connections', 'conn', 
                             stats.get('curr_connections'))
        if self.hasGraph('memcached_items'):
            self.setGraphVal('memcached_items', 'items', 
                             stats.get('curr_items'))
        if self.hasGraph('memcached_memory'):
            self.setGraphVal('memcached_memory', 'bytes', 
                             stats.get('bytes'))
        if self.hasGraph('memcached_connrate'):
            self.setGraphVal('memcached_connrate', 'conn', 
                             stats.get('total_connections'))
        if self.hasGraph('memcached_traffic'):
            self.setGraphVal('memcached_traffic', 'rxbytes', 
                             stats.get('bytes_read'))
            self.setGraphVal('memcached_traffic', 'txbytes', 
                             stats.get('bytes_written'))
        if self.hasGraph('memcached_reqrate'):
            self.setGraphVal('memcached_reqrate', 'set', 
                             stats.get('cmd_set'))
            self.setGraphVal('memcached_reqrate', 'get', 
                             stats.get('cmd_get'))
            if self.graphHasField('memcached_reqrate', 'del'):
                self.setGraphVal('memcached_reqrate', 'del',
                                 safe_sum([stats.get('delete_hits'), 
                                           stats.get('delete_misses')]))
            if self.graphHasField('memcached_reqrate', 'cas'):
                self.setGraphVal('memcached_reqrate', 'cas',
                                 safe_sum([stats.get('cas_hits'), 
                                           stats.get('cas_misses'), 
                                           stats.get('cas_badval')]))
            if self.graphHasField('memcached_reqrate', 'incr'):
                self.setGraphVal('memcached_reqrate', 'incr',
                                 safe_sum([stats.get('incr_hits'), 
                                           stats.get('incr_misses')]))
            if self.graphHasField('memcached_reqrate', 'decr'):
                self.setGraphVal('memcached_reqrate', 'decr',
                                 safe_sum([stats.get('decr_hits'), 
                                           stats.get('decr_misses')]))
        if self.hasGraph('memcached_statget'):
            self.setGraphVal('memcached_statget', 'hit', 
                             stats.get('get_hits'))
            self.setGraphVal('memcached_statget', 'miss', 
                             stats.get('get_misses'))
            self.setGraphVal('memcached_statget', 'total', 
                             safe_sum([stats.get('get_hits'),
                                       stats.get('get_misses')]))
        if self.hasGraph('memcached_statset'):
            self.setGraphVal('memcached_statset', 'hit', 
                             stats.get('set_hits'))
            self.setGraphVal('memcached_statset', 'miss', 
                             stats.get('set_misses'))
            self.setGraphVal('memcached_statset', 'total', 
                             safe_sum([stats.get('set_hits'),
                                       stats.get('set_misses')]))
        if self.hasGraph('memcached_statdel'):
            self.setGraphVal('memcached_statdel', 'hit', 
                             stats.get('delete_hits'))
            self.setGraphVal('memcached_statdel', 'miss', 
                             stats.get('delete_misses'))
            self.setGraphVal('memcached_statdel', 'total', 
                             safe_sum([stats.get('delete_hits'),
                                       stats.get('delete_misses')]))
        if self.hasGraph('memcached_statcas'):
            self.setGraphVal('memcached_statcas', 'hit', 
                             stats.get('cas_hits'))
            self.setGraphVal('memcached_statcas', 'miss', 
                             stats.get('cas_misses'))
            self.setGraphVal('memcached_statcas', 'badval', 
                             stats.get('cas_badval'))
            self.setGraphVal('memcached_statcas', 'total', 
                             safe_sum([stats.get('cas_hits'),
                                       stats.get('cas_misses'),
                                       stats.get('cas_badval')]))
        if self.hasGraph('memcached_statincrdecr'):
            self.setGraphVal('memcached_statincrdecr', 'incr_hit', 
                             stats.get('incr_hits'))
            self.setGraphVal('memcached_statincrdecr', 'decr_hit', 
                             stats.get('decr_hits'))
            self.setGraphVal('memcached_statincrdecr', 'incr_miss', 
                             stats.get('incr_misses'))
            self.setGraphVal('memcached_statincrdecr', 'decr_miss', 
                             stats.get('decr_misses'))
            self.setGraphVal('memcached_statincrdecr', 'total', 
                             safe_sum([stats.get('incr_hits'),
                                       stats.get('decr_hits'),
                                       stats.get('incr_misses'),
                                       stats.get('decr_misses')]))
        if self.hasGraph('memcached_statevict'):
            self.setGraphVal('memcached_statevict', 'evict', 
                             stats.get('evictions'))
            if self.graphHasField('memcached_statevict', 'reclaim'):
                self.setGraphVal('memcached_statevict', 'reclaim', 
                                 stats.get('reclaimed'))
        if self.hasGraph('memcached_statauth'):
            self.setGraphVal('memcached_statauth', 'reqs', 
                             stats.get('auth_cmds'))
            self.setGraphVal('memcached_statauth', 'errors', 
                             stats.get('auth_errors'))
        if self.hasGraph('memcached_hitpct'):
            prev_stats = self._prev_stats
            for (field_name,  field_hits,  field_misses) in (
                    ('set',  'set_hits',  'set_misses'),
                    ('get',  'get_hits',  'get_misses'), 
                    ('del',  'delete_hits',  'delete_misses'), 
                    ('cas',  'cas_hits',  'cas_misses'), 
                    ('incr',  'incr_hits',  'incr_misses'), 
                    ('decr',  'decr_hits',  'decr_misses')
                ):
                if prev_stats:
                    if (stats.has_key(field_hits) 
                        and prev_stats.has_key(field_hits)
                        and stats.has_key(field_misses) 
                        and prev_stats.has_key(field_misses)):
                        hits = stats[field_hits] - prev_stats[field_hits]
                        misses = stats[field_misses] - prev_stats[field_misses]
                        total = hits + misses
                        if total > 0:
                            val = 100.0 * hits / total
                        else:
                            val = 0
                        self.setGraphVal('memcached_hitpct',  field_name, 
                                         round(val,  2))