def retrieveVals(self):
        """Retrieve values for graphs."""
        if self._genStats is None:
            self._genStats = self._dbconn.getStats()
        if self._genVars is None:
            self._genVars = self._dbconn.getParams()
        if self.hasGraph('mysql_connections'):
            self.setGraphVal('mysql_connections', 'conn',
                             self._genStats.get('Connections'))
            self.setGraphVal('mysql_connections', 'abort_conn',
                             self._genStats.get('Aborted_connects'))
            self.setGraphVal('mysql_connections', 'abort_client',
                             self._genStats.get('Aborted_clients'))
        if self.hasGraph('mysql_traffic'):
            self.setGraphVal('mysql_traffic', 'rx',
                             self._genStats.get('Bytes_received'))
            self.setGraphVal('mysql_traffic', 'tx',
                             self._genStats.get('Bytes_sent'))
        if self.graphEnabled('mysql_slowqueries'):
            self.setGraphVal('mysql_slowqueries', 'queries',
                             self._genStats.get('Slow_queries'))
        if self.hasGraph('mysql_rowmodifications'):
            self.setGraphVal('mysql_rowmodifications', 'insert',
                             self._genStats.get('Handler_write'))
            self.setGraphVal('mysql_rowmodifications', 'update',
                             self._genStats.get('Handler_update'))
            self.setGraphVal('mysql_rowmodifications', 'delete',
                             self._genStats.get('Handler_delete'))
        if self.hasGraph('mysql_rowreads'):
            for field in self.getGraphFieldList('mysql_rowreads'):
                self.setGraphVal('mysql_rowreads', field, 
                                 self._genStats.get('Handler_read_%s' % field))
        if self.hasGraph('mysql_tablelocks'):
            self.setGraphVal('mysql_tablelocks', 'waited',
                             self._genStats.get('Table_locks_waited'))
            self.setGraphVal('mysql_tablelocks', 'immediate',
                             self._genStats.get('Table_locks_immediate'))
        if self.hasGraph('mysql_threads'):
            self.setGraphVal('mysql_threads', 'running',
                             self._genStats.get('Threads_running'))
            self.setGraphVal('mysql_threads', 'idle',
                             self._genStats.get('Threads_connected')
                             - self._genStats.get('Threads_running'))
            self.setGraphVal('mysql_threads', 'cached',
                             self._genStats.get('Threads_cached'))
            self.setGraphVal('mysql_threads', 'total',
                             self._genStats.get('Threads_connected') 
                             + self._genStats.get('Threads_cached'))
        if self.hasGraph('mysql_commits_rollbacks'):
            self.setGraphVal('mysql_commits_rollbacks', 'commit',
                             self._genStats.get('Handler_commit'))
            self.setGraphVal('mysql_commits_rollbacks', 'rollback',
                             self._genStats.get('Handler_rollback'))
        if self.hasGraph('mysql_qcache_memory'):
            try:
                total = self._genVars['query_cache_size']
                free = self._genStats['Qcache_free_memory']
                used = total - free
            except KeyError:
                free = None
                used = None
            self.setGraphVal('mysql_qcache_memory', 'used', used)
            self.setGraphVal('mysql_qcache_memory', 'free', free)
        if self.hasGraph('mysql_qcache_hits'):
            try:
                hits = self._genStats['Qcache_hits']
                misses = self._genStats['Com_select'] - hits
            except KeyError:
                hits = None
                misses = None
            self.setGraphVal('mysql_qcache_hits', 'hits', hits)
            self.setGraphVal('mysql_qcache_hits', 'misses', misses)
            
        if self.hasGraph('mysql_qcache_prunes'):
            self.setGraphVal('mysql_qcache_prunes', 'insert', 
                             self._genStats.get('Qcache_inserts'))
            self.setGraphVal('mysql_qcache_prunes', 'prune',
                             self._genStats.get('Qcache_lowmem_prunes'))
        if self.hasGraph('mysql_proc_status'):
            self._procStatus = self._dbconn.getProcessStatus()
            if self._procStatus:
                stats = {}
                for field in self.getGraphFieldList('mysql_proc_status'):
                    stats[field] = 0
                for (k, v) in self._procStatus.items():
                    if stats.has_key(k):
                        stats[k] = v
                    else:
                        stats['unknown'] += v
                for (k,v) in stats.items():
                    self.setGraphVal('mysql_proc_status', k, v)
        if self.hasGraph('mysql_proc_db'):
            self._procDB = self._dbconn.getProcessDatabase()
            for db in self._dbList:
                self.setGraphVal('mysql_proc_db', db, self._procDB.get(db, 0))
           
        if self.engineIncluded('myisam'):
            
            if self.hasGraph('mysql_myisam_key_buffer_util'):
                try:
                    bsize = self._genVars['key_cache_block_size']
                    total = self._genVars['key_buffer_size']
                    free = self._genStats['Key_blocks_unused'] * bsize
                    dirty = self._genStats['Key_blocks_not_flushed'] * bsize
                    clean = total - free - dirty
                except KeyError:
                    total = None
                    free = None
                    dirty = None
                    clean = None
                for (field,val) in (('dirty', dirty), 
                                    ('clean', clean),
                                    ('free', free),
                                    ('total', total)):
                    self.setGraphVal('mysql_myisam_key_buffer_util', 
                                     field, val)
            if self.hasGraph('mysql_myisam_key_read_reqs'):
                try:
                    misses = self._genStats['Key_reads']
                    hits = (self._genStats['Key_read_requests']
                            - misses)
                except KeyError:
                    misses = None
                    hits = None
                self.setGraphVal('mysql_myisam_key_read_reqs', 'disk', misses)
                self.setGraphVal('mysql_myisam_key_read_reqs', 'buffer', hits)
            
        if self.engineIncluded('innodb'):
            
            if self.hasGraph('mysql_innodb_buffer_pool_util'):
                self._genStats['Innodb_buffer_pool_pages_clean'] = (
                    self._genStats.get('Innodb_buffer_pool_pages_data')
                    - self._genStats.get('Innodb_buffer_pool_pages_dirty'))
                page_size = int(self._genStats.get('Innodb_page_size'))
                for field in ('dirty', 'clean', 'misc', 'free', 'total'):
                    self.setGraphVal('mysql_innodb_buffer_pool_util', 
                                     field, 
                                     self._genStats.get('Innodb_buffer_pool_pages_%s'
                                                        % field)
                                     * page_size)
            if self.hasGraph('mysql_innodb_buffer_pool_activity'):
                for field in ('created', 'read', 'written'):
                    self.setGraphVal('mysql_innodb_buffer_pool_activity', field, 
                                     self._genStats.get('Innodb_pages_%s' % field))
            if self.hasGraph('mysql_innodb_buffer_pool_read_reqs'):
                try:
                    misses = self._genStats['Innodb_buffer_pool_reads']
                    hits = (self._genStats['Innodb_buffer_pool_read_requests']
                            - misses)
                except KeyError:
                    misses = None
                    hits = None
                self.setGraphVal('mysql_innodb_buffer_pool_read_reqs', 'disk', 
                                 misses)
                self.setGraphVal('mysql_innodb_buffer_pool_read_reqs', 'buffer', 
                                 hits)
            if self.hasGraph('mysql_innodb_row_ops'):
                for field in ('inserted', 'updated', 'deleted', 'read'):
                    self.setGraphVal('mysql_innodb_row_ops', field, 
                                     self._genStats.get('Innodb_rows_%s' % field))