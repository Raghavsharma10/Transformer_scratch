def to_dict(self):
        """ Return a dictionary of the broker stats.

        Returns:
            dict: Dictionary of the stats.
        """
        return {
            'hostname': self.hostname,
            'port': self.port,
            'transport': self.transport,
            'virtual_host': self.virtual_host
        }