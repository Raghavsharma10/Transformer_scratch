def getPeerStats(self):
        """Get NTP Peer Stats for localhost by querying local NTP Server.
        
        @return: Dictionary of NTP stats converted to seconds.

        """
        info_dict = {}
        output = util.exec_command([ntpqCmd, '-n', '-c', 'peers'])
        for line in output.splitlines():
            mobj = re.match('\*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+', line)
            if mobj:
                info_dict['ip'] = mobj.group(1)
                cols = line.split()
                info_dict['stratum'] = int(cols[2])
                info_dict['delay'] = float(cols[7]) / 1000.0
                info_dict['offset'] = float(cols[8]) / 1000.0
                info_dict['jitter'] = float(cols[9]) / 1000.0
                return info_dict
        else:
            raise Exception("Execution of command failed: %s" % ntpqCmd)
        return info_dict