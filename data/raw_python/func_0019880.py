def getQueueStats(self):
        """Query Asterisk Manager Interface for Queue Stats.
        
        CLI Command: queue show
        
        @return: Dictionary of queue stats.
        
        """
        if not self.hasQueue():
            return None
        info_dict = {}
        if self.checkVersion('1.4'):
            cmd = "queue show"
        else:
            cmd = "show queues"
        cmdresp = self.executeCommand(cmd)
        
        queue = None
        ctxt = None
        member_states = ("unknown", "not in use", "in use", "busy", "invalid", 
                         "unavailable", "ringing", "ring+inuse", "on hold", 
                         "total")
        member_state_dict = dict([(k.lower().replace(' ', '_'),0) 
                                  for k in member_states]) 
        for line in cmdresp.splitlines():
            mobj = re.match(r"([\w\-]+)\s+has\s+(\d+)\s+calls\s+"
                            r"\(max (\d+|unlimited)\)\s+in\s+'(\w+)'\s+strategy\s+"
                            r"\((.+)\),\s+W:(\d+),\s+C:(\d+),\s+A:(\d+),\s+"
                            r"SL:([\d\.]+)%\s+within\s+(\d+)s", line)
            if mobj:
                ctxt = None
                queue = mobj.group(1)
                info_dict[queue] = {}
                info_dict[queue]['queue_len'] = int(mobj.group(2))
                try:
                    info_dict[queue]['queue_maxlen'] = int(mobj.group(3))
                except ValueError:
                    info_dict[queue]['queue_maxlen'] = None
                info_dict[queue]['strategy'] = mobj.group(4)
                for tkn in mobj.group(5).split(','):
                    mobjx = re.match(r"\s*(\d+)s\s+(\w+)\s*", tkn)
                    if mobjx:
                        info_dict[queue]['avg_' + mobjx.group(2)] = int(mobjx.group(1))
                info_dict[queue]['queue_weight'] = int(mobj.group(6))
                info_dict[queue]['calls_completed'] = int(mobj.group(7))
                info_dict[queue]['calls_abandoned'] = int(mobj.group(8))
                info_dict[queue]['sla_pcent'] = float(mobj.group(9))
                info_dict[queue]['sla_cutoff'] = int(mobj.group(10))
                info_dict[queue]['members'] = member_state_dict.copy() 
                continue
            mobj = re.match('\s+(Members|Callers):\s*$', line)
            if mobj:
                ctxt = mobj.group(1).lower()
                continue
            if ctxt == 'members':
                mobj = re.match(r"\s+\S.*\s\((.*)\)\s+has\s+taken.*calls", line)
                if mobj:
                    info_dict[queue]['members']['total'] += 1
                    state = mobj.group(1).lower().replace(' ', '_')
                    if info_dict[queue]['members'].has_key(state):
                        info_dict[queue]['members'][state] += 1
                    else:
                        raise AttributeError("Undefined queue member state %s"
                                             % state)
                    continue
        return info_dict