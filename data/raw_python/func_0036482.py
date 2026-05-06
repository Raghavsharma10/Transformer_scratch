def serialize(self):
        """
        Serializes the Peer data as a simple JSON map string.
        """
        return json.dumps({
            "name": self.name,
            "ip": self.ip,
            "port": self.port
        }, sort_keys=True)