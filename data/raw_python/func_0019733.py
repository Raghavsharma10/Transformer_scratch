def getMenu(self):
        """Get manager interface section list from Squid Proxy Server
        
        @return: List of tuples (section, description, type)
            
        """
        data = self._retrieve('')
        info_list = []
        for line in data.splitlines():
            mobj = re.match('^\s*(\S.*\S)\s*\t\s*(\S.*\S)\s*\t\s*(\S.*\S)$', line)
            if mobj:
                info_list.append(mobj.groups())
        return info_list