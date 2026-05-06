def retrieveVals(self):
        """Retrieve values for graphs."""
        proc_info = ProcessInfo()
        stats = {}
        for (prefix, is_thread) in (('proc', False), 
                                    ('thread', True)):
            graph_name = '%s_status' % prefix
            if self.hasGraph(graph_name):
                if not stats.has_key(prefix):
                    stats[prefix] = proc_info.getProcStatStatus(is_thread)
                for (fname, stat_key) in (
                    ('unint_sleep', 'uninterruptable_sleep'),
                    ('stopped', 'stopped'),
                    ('defunct', 'defunct'),
                    ('running', 'running'),
                    ('sleep', 'sleep')):
                    self.setGraphVal(graph_name, fname, 
                                     stats[prefix]['status'].get(stat_key))
            graph_name = '%s_prio' % prefix
            if self.hasGraph(graph_name):
                if not stats.has_key(prefix):
                    stats[prefix] = proc_info.getProcStatStatus(is_thread)
                for (fname, stat_key) in (
                    ('high', 'high'),
                    ('low', 'low'),
                    ('norm', 'norm'),
                    ('locked', 'locked_in_mem')):
                    self.setGraphVal(graph_name, fname, 
                                     stats[prefix]['prio'].get(stat_key))