def create(self, hostname, **kwargs):
        """
        Create new EC2 instance named ``hostname``.

        You may specify keyword arguments matching those of ``__init__`` (e.g.
        ``size``, ``ami``) to override any defaults given when the object was
        created, or to fill in parameters not given at initialization time.

        Additional parameters that are instance-specific:

        * ``ip``: The static private IP address for the new host.

        This method returns a ``boto.EC2.instance.Instance`` object.
        """
        # Create
        creating = "Creating '%s' (a %s instance of %s)" % (
            hostname, kwargs['size'], kwargs['ami']
        )
        with self.msg(creating):
            instance = self._create(hostname, kwargs)

        # Name
        with self.msg("Tagging as '%s'" % hostname):
            try:
                instance.rename(hostname)
            # One-time retry for API errors when setting tags
            except _ResponseError:
                time.sleep(1)
                instance.rename(hostname)

        # Wait for it to finish booting
        with self.msg("Waiting for boot: "):
            tick = 5
            while instance.state != 'running':
                self.log(".", end='')
                time.sleep(tick)
                instance.update()

        return instance