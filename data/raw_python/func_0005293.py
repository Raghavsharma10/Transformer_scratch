def add_slave(self, slave, container_name="widget"):
        """Add a slave delegate
        """
        cont = getattr(self, container_name, None)
        if cont is None:
            raise AttributeError(
                'Container name must be a member of the delegate')
        cont.add(slave.widget)
        self.slaves.append(slave)
        return slave