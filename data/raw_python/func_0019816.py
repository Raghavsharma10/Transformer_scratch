def getIfConfig(self):
        """Return dictionary of Interface Configuration (ifconfig).
        
        @return: Dictionary of if configurations keyed by if name.
        
        """
        conf = {}
        try:
            out = subprocess.Popen([ipCmd, "addr", "show"], 
                                   stdout=subprocess.PIPE).communicate()[0]
        except:
            raise Exception('Execution of command %s failed.' % ipCmd)
        for line in out.splitlines():
            mobj = re.match('^\d+: (\S+):\s+<(\S*)>\s+(\S.*\S)\s*$', line)
            if mobj:
                iface = mobj.group(1)
                conf[iface] = {}
                continue
            mobj = re.match('^\s{4}link\/(.*\S)\s*$', line)
            if mobj:
                arr = mobj.group(1).split()
                if len(arr) > 0:
                    conf[iface]['type'] = arr[0]
                if len(arr) > 1:
                    conf[iface]['hwaddr'] = arr[1]
                continue
            mobj = re.match('^\s+(inet|inet6)\s+([\d\.\:A-Za-z]+)\/(\d+)($|\s+.*\S)\s*$', line)
            if mobj:
                proto = mobj.group(1)
                if not conf[iface].has_key(proto):
                    conf[iface][proto] = []
                addrinfo = {}
                addrinfo['addr'] = mobj.group(2).lower()
                addrinfo['mask'] = int(mobj.group(3))
                arr = mobj.group(4).split()
                if len(arr) > 0 and arr[0] == 'brd':
                    addrinfo['brd'] = arr[1]
                conf[iface][proto].append(addrinfo)
                continue
        return conf