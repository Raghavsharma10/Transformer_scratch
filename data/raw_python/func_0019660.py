def retrieveVals(self):
        """Retrieve values for graphs."""
        if self.hasGraph('tomcat_memory'):
            stats = self._tomcatInfo.getMemoryStats()
            self.setGraphVal('tomcat_memory', 'used', 
                             stats['total'] - stats['free'])
            self.setGraphVal('tomcat_memory', 'free', stats['free'])
            self.setGraphVal('tomcat_memory', 'max', stats['max'])
        for (port, stats) in self._tomcatInfo.getConnectorStats().iteritems():
            thrstats = stats['threadInfo']
            reqstats = stats['requestInfo']
            if self.portIncluded(port):
                name = "tomcat_threads_%d" % port
                if self.hasGraph(name):
                    self.setGraphVal(name, 'busy', 
                                     thrstats['currentThreadsBusy'])
                    self.setGraphVal(name, 'idle', 
                                     thrstats['currentThreadCount'] 
                                     - thrstats['currentThreadsBusy'])
                    self.setGraphVal(name, 'max', thrstats['maxThreads'])
                name = "tomcat_access_%d" % port
                if self.hasGraph(name):
                    self.setGraphVal(name, 'reqs', reqstats['requestCount'])
                name = "tomcat_error_%d" % port
                if self.hasGraph(name):
                    self.setGraphVal(name, 'errors', reqstats['errorCount'])
                name = "tomcat_traffic_%d" % port
                if self.hasGraph(name):
                    self.setGraphVal(name, 'rx', reqstats['bytesReceived'])
                    self.setGraphVal(name, 'tx', reqstats['bytesSent'])