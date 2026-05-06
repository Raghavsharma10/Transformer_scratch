def getFaxStatsSessions(self):
        """Query Asterisk Manager Interface for Fax Stats.
        
        CLI Command - fax show sessions
        
        @return: Dictionary of fax stats.
        
        """
        if not self.hasFax():
            return None
        info_dict = {}
        info_dict['total'] = 0
        fax_types = ('g.711', 't.38')
        fax_operations = ('send', 'recv')
        fax_states = ('uninitialized', 'initialized', 'open', 
                      'active', 'inactive', 'complete', 'unknown',)
        info_dict['type'] = dict([(k,0) for k in fax_types])
        info_dict['operation'] = dict([(k,0) for k in fax_operations])
        info_dict['state'] = dict([(k,0) for k in fax_states])
        cmdresp = self.executeCommand('fax show sessions')
        sections = cmdresp.strip().split('\n\n')
        if len(sections) >= 3:
            for line in sections[1][1:]:
                cols = re.split('\s\s+', line)
                if len(cols) == 7:
                    info_dict['total'] += 1
                    if cols[3].lower() in fax_types:
                        info_dict['type'][cols[3].lower()] += 1
                    if cols[4] == 'receive':
                        info_dict['operation']['recv'] += 1
                    elif cols[4] == 'send':
                        info_dict['operation']['send'] += 1
                    if cols[5].lower() in fax_states:
                        info_dict['state'][cols[5].lower()] += 1
        return info_dict