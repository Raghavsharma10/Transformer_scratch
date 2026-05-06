def loadJson(self, data):
        """convert the json data into updating this obj's attrs"""
        if not isinstance(data, dict):
            data = json.loads(data)
        self.__dict__.update(data)
        self.inflate() # restore objects from str values
        #if self.ports:  self._gotPorts = True
        return self