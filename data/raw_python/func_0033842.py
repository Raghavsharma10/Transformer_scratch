def find_types(self, site=None, match=None):
        """Query the LDR host for frame types. Use site to restrict
        query to given observatory prefix, and use match to restrict
        returned types to those matching the regular expression.

        Example:

        >>> connection.find_types("L", "RDS")
        ['L1_RDS_C01_LX',
         'L1_RDS_C02_LX',
         'L1_RDS_C03_L2',
         'L1_RDS_R_L1',
         'L1_RDS_R_L3',
         'L1_RDS_R_L4',
         'PEM_RDS_A6',
         'RDS_R_L1',
         'RDS_R_L2',
         'RDS_R_L3',
         'TESTPEM_RDS_A6']

        @param  site: single-character name of site to match
        @param match: type-name to match against

        @type  site: L{str}
        @type match: L{str}

        @returns: L{list} of frame types
        """
        if site:
            url = "%s/gwf/%s.json" % (_url_prefix, site[0])
        else:
            url = "%s/gwf/all.json" % _url_prefix
        response = self._requestresponse("GET", url)
        typelist = sorted(set(decode(response.read())))
        if match:
            regmatch = re.compile(match)
            typelist = [type for type in typelist if regmatch.search(type)]
        return typelist