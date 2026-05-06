def modules(self):
        """
        :return: dictionary {index: object} of all modules.
        """

        if not self.get_objects_by_type('module'):
            self.inventory()
        return {int(c.index): c for c in self.get_objects_by_type('module')}