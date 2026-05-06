def getconnections(self, vhost = None):
        "Return accepted connections, optionally filtered by vhost"
        if vhost is None:
            return list(self.managed_connections)
        else:
            return [c for c in self.managed_connections if c.protocol.vhost == vhost]