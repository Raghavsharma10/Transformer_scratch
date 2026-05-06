def _reset(self, **kwargs):
        """
        Reset the objects attributes.

        Accepts servers as either unflattened or flattened UUID strings or Server objects.
        """
        super(Tag, self)._reset(**kwargs)

        # backup name for changing it (look: Tag.save)
        self._api_name = self.name

        # flatten { servers: { server: [] } }
        if 'server' in self.servers:
            self.servers = kwargs['servers']['server']

        # convert UUIDs into server objects
        if self.servers and isinstance(self.servers[0], six.string_types):
            self.servers = [Server(uuid=server, populated=False) for server in self.servers]