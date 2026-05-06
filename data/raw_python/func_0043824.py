def returnPorts(self):
        """deallocate specific ports on the current machine"""
        if self._gotPorts:
            #print("deleting ports >%s<"%(self.ports))
            map(portpicker.return_port, self.ports)
            self._gotPorts = False
        self.ports = []