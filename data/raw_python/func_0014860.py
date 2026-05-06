def clients(self):
        """
        Get all clients (with and without associated resources)
        """
        clients = {}
        for k, v in self.connections.items():
            if hasattr(v.meta, 'client'):       # has boto3 resource
                clients[k] = v.meta.client
            else:                               # no boto3 resource
                clients[k] = v
        return clients