def get_services(self):
        """Returns a list of FritzService-objects."""
        result = []
        nodes = self.root.iterfind(
            './/ns:service', namespaces={'ns': self.namespace})
        for node in nodes:
            result.append(FritzService(
                node.find(self.nodename('serviceType')).text,
                node.find(self.nodename('controlURL')).text,
                node.find(self.nodename('SCPDURL')).text))
        return result