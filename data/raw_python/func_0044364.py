def create_waf(self, name, waf_type):
        """
        Creates a WAF with the given type.
        :param name: Name of the WAF.
        :param waf_type: WAF type. ('mod_security', 'Snort', 'Imperva SecureSphere', 'F5 BigIP ASM', 'DenyAll rWeb')
        """
        params = {
            'name': name,
            'type': waf_type
        }
        return self._request('POST', 'rest/wafs/new', params)