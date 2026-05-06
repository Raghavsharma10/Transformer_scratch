def getRoutes(self):
        """Get routing table.
        
        @return: List of routes.
        
        """
        routes = []
        try:
            out = subprocess.Popen([routeCmd, "-n"], 
                                   stdout=subprocess.PIPE).communicate()[0]
        except:
            raise Exception('Execution of command %s failed.' % ipCmd)
        lines = out.splitlines()
        if len(lines) > 1:
            headers = [col.lower() for col in lines[1].split()]
            for line in lines[2:]:
                routes.append(dict(zip(headers, line.split())))
        return routes