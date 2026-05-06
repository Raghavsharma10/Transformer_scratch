def serialize(self):
        """
        Serializes the node data as a JSON map string.
        """
        return json.dumps({
            "port": self.port,
            "ip": self.ip,
            "host": self.host,
            "peer": self.peer.serialize() if self.peer else None,
            "metadata": json.dumps(self.metadata or {}, sort_keys=True),
        }, sort_keys=True)