def getStats(self, tcp=True, udp=True, ipv4=True, ipv6=True, 
                 include_listen=True, only_listen=False,
                 show_users=False, show_procs=False, 
                 resolve_hosts=False, resolve_ports=False, resolve_users=True, 
                 **kwargs):
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
        @param **kwargs:       Keyword variables are used for filtering the 
                               results depending on the values of the columns. 
                               Each keyword must correspond to a field name with 
                               an optional suffix:
                               field:          Field equal to value or in list 
                                               of values.
                               field_ic:       Field equal to value or in list of 
                                               values, using case insensitive 
                                               comparison.
                               field_regex:    Field matches regex value or 
                                               matches with any regex in list of 
                                               values.
                               field_ic_regex: Field matches regex value or 
                                               matches with any regex in list of 
                                               values using case insensitive 
                                               match.
        @return:               List of headers and list of rows and columns.
        
        """
        pinfo = self.parseNetstatCmd(tcp, udp, ipv4, ipv6, 
                                     include_listen, only_listen,
                                     show_users, show_procs, 
                                     resolve_hosts, resolve_ports, resolve_users)
        if pinfo:
            if len(kwargs) > 0:
                pfilter = util.TableFilter()
                pfilter.registerFilters(**kwargs)
                stats = pfilter.applyFilters(pinfo['headers'], pinfo['stats'])
                return {'headers': pinfo['headers'], 'stats': stats}
            else:
                return pinfo
        else:
            return None