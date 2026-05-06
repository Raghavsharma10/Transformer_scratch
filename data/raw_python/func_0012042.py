def node(self,port, hub_address=("localhost", 4444)):
        ''' java -jar selenium-server.jar -role node -port 5555 -hub http://127.0.0.1:4444/grid/register/
        @param port:  listen port of selenium node
        @param hub_address: hub address which node will connect to 
        '''
        self._ip, self._port = hub_address
        self.command = [self._conf["java_path"], "-jar", self._conf["jar_path"], "-port", str(port), "-role", "node", "-hub", "http://%s:%s/grid/register/" %(self._ip, self._port)]        
        return self