def retrieveVals(self):
        """Retrieve values for graphs."""                
        stats = self._dbconn.getDatabaseStats()
        databases = stats.get('databases')
        totals = stats.get('totals')
        if self.hasGraph('pg_connections'):
            limit = self._dbconn.getParam('max_connections')
            self.setGraphVal('pg_connections', 'max_conn', limit)
            for (db, dbstats) in databases.iteritems():
                if self.dbIncluded(db):
                    self.setGraphVal('pg_connections', db, 
                                     dbstats['numbackends'])
            self.setGraphVal('pg_connections', 'total', totals['numbackends'])
        if self.hasGraph('pg_diskspace'):
            for (db, dbstats) in databases.iteritems():
                if self.dbIncluded(db):
                    self.setGraphVal('pg_diskspace', db, dbstats['disk_size'])
            self.setGraphVal('pg_diskspace', 'total', totals['disk_size'])
        if self.hasGraph('pg_blockreads'):
            self.setGraphVal('pg_blockreads', 'blk_hit', totals['blks_hit'])
            self.setGraphVal('pg_blockreads', 'blk_read', totals['blks_read'])
        if self.hasGraph('pg_xact'):
            self.setGraphVal('pg_xact', 'commits', totals['xact_commit'])
            self.setGraphVal('pg_xact', 'rollbacks', totals['xact_rollback'])
        if self.hasGraph('pg_tup_read'):
            self.setGraphVal('pg_tup_read', 'fetch', totals['tup_fetched'])
            self.setGraphVal('pg_tup_read', 'return', totals['tup_returned'])
        if self.hasGraph('pg_tup_write'):
            self.setGraphVal('pg_tup_write', 'delete', totals['tup_deleted'])
            self.setGraphVal('pg_tup_write', 'update', totals['tup_updated'])
            self.setGraphVal('pg_tup_write', 'insert', totals['tup_inserted'])
        lock_stats = None
        for lock_state in ('all', 'wait',):
            graph_name = "pg_lock_%s" % lock_state
            if self.hasGraph(graph_name):
                if lock_stats is None:
                    lock_stats = self._dbconn.getLockStatsMode()
                mode_iter = iter(PgInfo.lockModes)
                for mode in ('AccessExcl', 'Excl', 'ShrRwExcl', 'Shr', 
                             'ShrUpdExcl', 'RwExcl', 'RwShr', 'AccessShr',):
                    self.setGraphVal(graph_name, mode, 
                                     lock_stats[lock_state].get(mode_iter.next()))
        
        stats = None               
        if self.hasGraph('pg_checkpoints'):
            if stats is None:
                stats = self._dbconn.getBgWriterStats()
            self.setGraphVal('pg_checkpoints', 'req', 
                             stats.get('checkpoints_req'))
            self.setGraphVal('pg_checkpoints', 'timed', 
                             stats.get('checkpoints_timed'))
        if self.hasGraph('pg_bgwriter'):
            if stats is None:
                stats = self._dbconn.getBgWriterStats()
            self.setGraphVal('pg_bgwriter', 'backend', 
                             stats.get('buffers_backend'))
            self.setGraphVal('pg_bgwriter', 'clean', 
                             stats.get('buffers_clean'))
            self.setGraphVal('pg_bgwriter', 'chkpoint', 
                             stats.get('buffers_checkpoint'))
            
        if self._detailGraphs:
            for (db, dbstats) in databases.iteritems():
                if self.dbIncluded(db):
                    if self.hasGraph('pg_blockread_detail'):
                        self.setGraphVal('pg_blockread_detail', db, 
                            dbstats['blks_hit'] + dbstats['blks_read'])
                    for (graph_name, attr_name) in (
                            ('pg_xact_commit_detail', 'xact_commit'),
                            ('pg_xact_rollback_detail', 'xact_rollback'),
                            ('pg_tup_return_detail', 'tup_returned'),
                            ('pg_tup_fetch_detail', 'tup_fetched'),
                            ('pg_tup_delete_detail', 'tup_deleted'),
                            ('pg_tup_update_detail', 'tup_updated'),
                            ('pg_tup_insert_detail', 'tup_inserted'),
                        ):
                        if self.hasGraph(graph_name):
                            self.setGraphVal(graph_name, db, dbstats[attr_name])
                    lock_stats_db = None
                    for lock_state in ('all', 'wait',):
                        graph_name = "pg_lock_%s_detail" % lock_state
                        if self.hasGraph(graph_name):
                            if lock_stats_db is None:
                                lock_stats_db = self._dbconn.getLockStatsDB()
                            self.setGraphVal(graph_name, db, 
                                             lock_stats_db[lock_state].get(db, 0))
            
        if self._replGraphs:
            repl_stats = self._dbconn.getSlaveConflictStats()
            if self.hasGraph('pg_repl_conflicts'):        
                for field in self.getGraphFieldList('pg_repl_conflicts'):
                    self.setGraphVal('pg_repl_conflicts', field, 
                                     repl_stats['totals'].get("confl_%s" % field))
            if self._detailGraphs and self.hasGraph('pg_repl_conflicts_detail'):
                for (db, dbstats) in repl_stats['databases'].iteritems():
                    if self.dbIncluded(db):
                        self.setGraphVal('pg_repl_conflicts_detail', db,
                                         sum(dbstats.values()))