def ports(self):
        """
        :return: dictionary {index: object} of all ports.
        """

        if not self.get_objects_by_type('port'):
            self.inventory()
        return {int(p.index.split('/')[1]): p for p in self.get_objects_by_type('port')}