def hub(self, port):
        ''' java -jar selenium-server.jar -role hub -port 4444
        @param port:  listen port of selenium hub 
        '''
        self._ip = "localhost"
        self._port = port 
        self.command = [self._conf["java_path"], "-jar", self._conf["jar_path"], "-port", str(port), "-role", "hub"]        
        return self