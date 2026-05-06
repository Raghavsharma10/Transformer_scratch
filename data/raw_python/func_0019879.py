def getTrunkStats(self, trunkList):
        """Query Asterisk Manager Interface for Trunk Stats.
        
        CLI Command - core show channels

        @param trunkList: List of tuples of one of the two following types:
                            (Trunk Name, Regular Expression)
                            (Trunk Name, Regular Expression, MIN, MAX)
        @return: Dictionary of trunk utilization statistics.

        """
        re_list = []
        info_dict = {}
        for filt in trunkList:
            info_dict[filt[0]] = 0
            re_list.append(re.compile(filt[1], re.IGNORECASE))
                  
        if self.checkVersion('1.4'):
            cmd = "core show channels"
        else:
            cmd = "show channels"
        cmdresp = self.executeCommand(cmd)

        for line in cmdresp.splitlines():
            for idx in range(len(re_list)):
                recomp = re_list[idx]
                trunkid = trunkList[idx][0]
                mobj = recomp.match(line)
                if mobj:
                    if len(trunkList[idx]) == 2:
                        info_dict[trunkid] += 1
                        continue
                    elif len(trunkList[idx]) == 4:
                        num = mobj.groupdict().get('num')
                        if num is not None:
                            (vmin,vmax) = trunkList[idx][2:4]
                            if int(num) >= int(vmin) and int(num) <= int(vmax):
                                info_dict[trunkid] += 1
                                continue
        return info_dict