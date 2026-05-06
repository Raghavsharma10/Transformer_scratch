def rr_history(self, query, query_type="A"):
        '''Get the RR (Resource Record) History of the given domain or IP.
        The default query type is for 'A' records, but the following query types
        are supported:

        A, NS, MX, TXT, CNAME

        For details, see https://investigate.umbrella.com/docs/api#dnsrr_domain
        '''
        if query_type not in Investigate.SUPPORTED_DNS_TYPES:
            raise Investigate.UNSUPPORTED_DNS_QUERY

        # if this is an IP address, query the IP
        if Investigate.IP_PATTERN.match(query):
            return self._ip_rr_history(query, query_type)

        # otherwise, query the domain
        return self._domain_rr_history(query, query_type)