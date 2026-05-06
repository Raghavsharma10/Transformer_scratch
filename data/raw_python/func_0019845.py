def getMemoryStats(self):
        """Return JVM Memory Stats for Apache Tomcat Server.
        
        @return: Dictionary of memory utilization stats.
        
        """
        if self._statusxml is None:
            self.initStats()
        node = self._statusxml.find('jvm/memory')
        memstats = {}
        if node is not None:
            for (key,val) in node.items():
                memstats[key] = util.parse_value(val)
        return memstats