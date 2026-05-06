def parseNetstatCmd(self, tcp=True, udp=True, ipv4=True, ipv6=True, 
                        include_listen=True, only_listen=False,
                        show_users=False, show_procs=False,
                        resolve_hosts=False, resolve_ports=False, 
                        resolve_users=True):
        """Execute netstat command and return result as a nested dictionary.
        
        @param tcp:            Include TCP ports in ouput if True.
        @param udp:            Include UDP ports in ouput if True.
        @param ipv4:           Include IPv4 ports in output if True.
        @param ipv6:           Include IPv6 ports in output if True.
        @param include_listen: Include listening ports in output if True.
        @param only_listen:    Include only listening ports in output if True.
        @param show_users:     Show info on owning users for ports if True.
        @param show_procs:     Show info on PID and Program Name attached to
                               ports if True.
        @param resolve_hosts:  Resolve IP addresses into names if True.
        @param resolve_ports:  Resolve numeric ports to names if True.
        @param resolve_users:  Resolve numeric user IDs to user names if True.
        @return:               List of headers and list of rows and columns.
        
        """
        headers = ['proto', 'ipversion', 'recvq', 'sendq', 
                   'localaddr', 'localport','foreignaddr', 'foreignport', 
                   'state']
        args = []
        proto = []
        if ipv4:
            proto.append('inet')
        if ipv6:
            proto.append('inet6')
        if len(proto) > 0:
            args.append('-A')
            args.append(','.join(proto))
        if tcp:
            args.append('-t')
        if udp:
            args.append('-u')
        if only_listen:
            args.append('-l')
        elif include_listen:
            args.append('-a')
        regexp_str = ('(tcp|udp)(\d*)\s+(\d+)\s+(\d+)\s+'
                      '(\S+):(\w+)\s+(\S+):(\w+|\*)\s+(\w*)')
        if show_users:
            args.append('-e')
            regexp_str += '\s+(\w+)\s+(\d+)'
            headers.extend(['user', 'inode'])
        if show_procs:
            args.append('-p')
            regexp_str += '\s+(\S+)'
            headers.extend(['pid', 'prog'])
        if not resolve_hosts:
            args.append('--numeric-hosts')
        if not resolve_ports:
            args.append('--numeric-ports')
        if not resolve_users:
            args.append('--numeric-users')
        lines = self.execNetstatCmd(*args)
        stats = []
        regexp = re.compile(regexp_str)
        for line in lines[2:]:
            mobj = regexp.match(line)
            if mobj is not None:
                stat = list(mobj.groups())
                if stat[1] == '0':
                    stat[1] = '4'
                if stat[8] == '':
                    stat[8] = None
                if show_procs:
                    proc = stat.pop().split('/')
                    if len(proc) == 2:
                        stat.extend(proc)
                    else:
                        stat.extend([None, None])
                stats.append(stat)
        return {'headers': headers, 'stats': stats}