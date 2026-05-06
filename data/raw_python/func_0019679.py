def retrieveVals(self):
        """Retrieve values for graphs."""
        if self.hasGraph('sys_loadavg'):
            self._loadstats = self._sysinfo.getLoadAvg()
            if self._loadstats:
                self.setGraphVal('sys_loadavg', 'load15min', self._loadstats[2])
                self.setGraphVal('sys_loadavg', 'load5min', self._loadstats[1])
                self.setGraphVal('sys_loadavg', 'load1min', self._loadstats[0])
        if self._cpustats and self.hasGraph('sys_cpu_util'):
            for field in self.getGraphFieldList('sys_cpu_util'):
                self.setGraphVal('sys_cpu_util', 
                                 field, int(self._cpustats[field] * 1000))
        if self._memstats:
            if self.hasGraph('sys_mem_util'):
                for field in self.getGraphFieldList('sys_mem_util'):
                    self.setGraphVal('sys_mem_util', 
                                     field, self._memstats[field])
            if self.hasGraph('sys_mem_avail'):
                for field in self.getGraphFieldList('sys_mem_avail'):
                    self.setGraphVal('sys_mem_avail', 
                                     field, self._memstats[field])
            if self.hasGraph('sys_mem_huge'):
                for field in ['Rsvd', 'Surp', 'Free']:
                    fkey = 'HugePages_' + field
                    if self._memstats.has_key(fkey):
                        self.setGraphVal('sys_mem_huge', field, 
                            self._memstats[fkey] * self._memstats['Hugepagesize'])
        if self.hasGraph('sys_processes'):
            if self._procstats is None:
                self._procstats = self._sysinfo.getProcessStats()
            if self._procstats:
                self.setGraphVal('sys_processes', 'running', 
                                 self._procstats['procs_running'])
                self.setGraphVal('sys_processes', 'blocked', 
                                 self._procstats['procs_blocked'])
        if self.hasGraph('sys_forks'):
            if self._procstats is None:
                self._procstats = self._sysinfo.getProcessStats()
            if self._procstats:
                self.setGraphVal('sys_forks', 'forks', 
                                 self._procstats['processes'])
        if self.hasGraph('sys_intr_ctxt'):
            if self._procstats is None:
                self._procstats = self._sysinfo.getProcessStats()
            if self._procstats:
                for field in self.getGraphFieldList('sys_intr_ctxt'):
                    self.setGraphVal('sys_intr_ctxt', field, 
                                     self._procstats[field])
        if self.hasGraph('sys_vm_paging'):
            if self._vmstats is None:
                self._vmstats = self._sysinfo.getVMstats()
            if self._vmstats:
                self.setGraphVal('sys_vm_paging', 'in', 
                                 self._vmstats['pgpgin'])
                self.setGraphVal('sys_vm_paging', 'out', 
                                 self._vmstats['pgpgout'])
        if self.hasGraph('sys_vm_swapping'):
            if self._vmstats is None:
                self._vmstats = self._sysinfo.getVMstats()
            if self._vmstats:
                self.setGraphVal('sys_vm_swapping', 'in', 
                                 self._vmstats['pswpin'])
                self.setGraphVal('sys_vm_swapping', 'out', 
                                 self._vmstats['pswpout'])