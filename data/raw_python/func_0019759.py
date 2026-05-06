def getHostOffsets(self, hosts):
        """Get NTP Stats and offset of multiple remote hosts relative to localhost
        by querying NTP Servers on remote hosts.
        
        @param host: List of Remote Host IPs.
        @return:     Dictionary of NTP stats converted to seconds.

        """
        info_dict = {}
        output = util.exec_command([ntpdateCmd, '-u', '-q'] + list(hosts))
        for line in output.splitlines():
            mobj = re.match('server\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}),'
                            '\s*stratum\s+(\d),.*offset\s+([\d\.-]+),'
                            '.*delay\s+([\d\.]+)\s*$', line)
            if mobj:
                host_dict = {}
                host = mobj.group(1)
                host_dict['stratum'] = int(mobj.group(2))
                host_dict['delay'] = float(mobj.group(4))
                host_dict['offset'] = float(mobj.group(3))
                info_dict[host] = host_dict
        return info_dict