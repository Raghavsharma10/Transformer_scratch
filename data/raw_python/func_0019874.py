def getChannelStats(self, chantypes=('dahdi', 'zap', 'sip', 'iax2', 'local')):
        """Query Asterisk Manager Interface for Channel Stats.
        
        CLI Command - core show channels

        @return: Dictionary of statistics counters for channels.
            Number of active channels for each channel type.

        """
        if self.checkVersion('1.4'):
            cmd = "core show channels"
        else:
            cmd = "show channels"
        cmdresp = self.executeCommand(cmd)
        info_dict ={}
        for chanstr in chantypes:
            chan = chanstr.lower()
            if chan in ('zap', 'dahdi'):
                info_dict['dahdi'] = 0
                info_dict['mix'] = 0
            else:
                info_dict[chan] = 0
        for k in ('active_calls', 'active_channels', 'calls_processed'):
            info_dict[k] = 0
        regexstr = ('(%s)\/(\w+)' % '|'.join(chantypes))    
        for line in cmdresp.splitlines():
            mobj = re.match(regexstr, 
                            line, re.IGNORECASE)
            if mobj:
                chan_type = mobj.group(1).lower()
                chan_id = mobj.group(2).lower()
                if chan_type == 'dahdi' or chan_type == 'zap':
                    if chan_id == 'pseudo':
                        info_dict['mix'] += 1
                    else:
                        info_dict['dahdi'] += 1
                else:
                    info_dict[chan_type] += 1
                continue

            mobj = re.match('(\d+)\s+(active channel|active call|calls processed)', 
                            line, re.IGNORECASE)
            if mobj:
                if mobj.group(2) == 'active channel':
                    info_dict['active_channels'] = int(mobj.group(1))
                elif mobj.group(2) == 'active call':
                    info_dict['active_calls'] = int(mobj.group(1))
                elif mobj.group(2) == 'calls processed':
                    info_dict['calls_processed'] = int(mobj.group(1))
                continue

        return info_dict