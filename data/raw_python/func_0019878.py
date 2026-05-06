def getVoicemailStats(self):
        """Query Asterisk Manager Interface for Voicemail Stats.
        
        CLI Command - voicemail show users

        @return: Dictionary of statistics counters for Voicemail Accounts.

        """
        if not self.hasVoicemail():
            return None
        if self.checkVersion('1.4'):
            cmd = "voicemail show users"
        else:
            cmd = "show voicemail users"
        cmdresp = self.executeCommand(cmd)

        info_dict = dict(accounts = 0, avg_messages = 0, max_messages = 0, 
                         total_messages = 0)
        for line in cmdresp.splitlines():
            mobj = re.match('\w+\s+\w+\s+.*\s+(\d+)\s*$', line)
            if mobj:
                msgs = int(mobj.group(1))
                info_dict['accounts'] += 1
                info_dict['total_messages'] += msgs
                if msgs > info_dict['max_messages']:
                    info_dict['max_messages'] = msgs
        if info_dict['accounts'] > 0:
            info_dict['avg_messages'] = (float(info_dict['total_messages']) 
                                         / info_dict['accounts'])
            
        return info_dict