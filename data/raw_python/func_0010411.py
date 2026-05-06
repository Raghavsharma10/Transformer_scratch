def _read_state_variables(self):
        """
        Reads the stateVariable information from the xml-file.
        The information we like to extract are name and dataType so we
        can assign them later on to FritzActionArgument-instances.
        Returns a dictionary: key:value = name:dataType
        """
        nodes = self.root.iterfind(
            './/ns:stateVariable', namespaces={'ns': self.namespace})
        for node in nodes:
            key = node.find(self.nodename('name')).text
            value = node.find(self.nodename('dataType')).text
            self.state_variables[key] = value