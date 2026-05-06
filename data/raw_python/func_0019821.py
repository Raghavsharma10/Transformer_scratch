def getTCPportConnStatus(self, ipv4=True, ipv6=True, include_listen=False,
                             **kwargs):
        """Returns the number of TCP endpoints discriminated by status.
        
        @param ipv4:           Include IPv4 ports in output if True.
        @param ipv6:           Include IPv6 ports in output if True.
        @param include_listen: Include listening ports in output if True.
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
        @return:               Dictionary mapping connection status to the
                               number of endpoints.
        
        """
        status_dict = {}
        result = self.getStats(tcp=True, udp=False, 
                               include_listen=include_listen, 
                               ipv4=ipv4, ipv6=ipv6, 
                               **kwargs)
        stats = result['stats']
        for stat in stats:
            if stat is not None:
                status = stat[8].lower()
            status_dict[status] = status_dict.get(status, 0) + 1
        return status_dict