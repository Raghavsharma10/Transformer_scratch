def getTCPportConnCount(self, ipv4=True, ipv6=True, resolve_ports=False,
                            **kwargs):
        """Returns TCP connection counts for each local port.
        
        @param ipv4:          Include IPv4 ports in output if True.
        @param ipv6:          Include IPv6 ports in output if True.
        @param resolve_ports: Resolve numeric ports to names if True.
        @param **kwargs:      Keyword variables are used for filtering the 
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
        @return:              Dictionary mapping port number or name to the
                              number of established connections.
        
        """
        port_dict = {}
        result = self.getStats(tcp=True, udp=False, 
                               include_listen=False, ipv4=ipv4, 
                               ipv6=ipv6, resolve_ports=resolve_ports,
                               **kwargs)
        stats = result['stats']
        for stat in stats:
            if stat[8] == 'ESTABLISHED':
                port_dict[stat[5]] = port_dict.get(5, 0) + 1
        return port_dict