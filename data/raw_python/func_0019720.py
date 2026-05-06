def getDevType(self, dev):
        """Returns type of device dev.
        
        @return: Device type as string.
        
        """
        if self._devClassTree is None:
            self._initDevClasses()
        return self._mapDevType.get(dev)