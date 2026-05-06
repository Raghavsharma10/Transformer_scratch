def getConferenceStats(self):
        """Query Asterisk Manager Interface for Conference Room Stats.
        
        CLI Command - meetme list

        @return: Dictionary of statistics counters for Conference Rooms.

        """
        if not self.hasConference():
            return None
        if self.checkVersion('1.6'):
            cmd = "meetme list"
        else:
            cmd = "meetme"
        cmdresp = self.executeCommand(cmd)

        info_dict = dict(active_conferences = 0, conference_users = 0)
        for line in cmdresp.splitlines():
            mobj = re.match('\w+\s+0(\d+)\s', line)
            if mobj:
                info_dict['active_conferences'] += 1
                info_dict['conference_users'] += int(mobj.group(1))

        return info_dict