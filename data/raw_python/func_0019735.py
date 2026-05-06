def getPRIstats(self, iface):
        """Return RDSI Operational Stats for interface.
        
        @param iface: Interface name. (Ex. w1g1)
        @return:      Nested dictionary of statistics for interface.

        """
        info_dict = {}
        output = util.exec_command([wanpipemonCmd, '-i', iface, '-c',  'Ta'])
        for line in output.splitlines():
            mobj = re.match('^\s*(Line Code Violation|Far End Block Errors|'
                            'CRC4 Errors|FAS Errors)\s*:\s*(\d+)\s*$', 
                            line, re.IGNORECASE)
            if mobj:
                info_dict[mobj.group(1).lower().replace(' ', '')] = int(mobj.group(2))
                continue
            mobj = re.match('^\s*(Rx Level)\s*:\s*>{0,1}\s*([-\d\.]+)db\s*', 
                            line, re.IGNORECASE)
            if mobj:
                info_dict[mobj.group(1).lower().replace(' ', '')] = float(mobj.group(2))
                continue
        return info_dict