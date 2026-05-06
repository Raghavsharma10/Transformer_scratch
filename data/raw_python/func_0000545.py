def inventory(self):
        """ Get inventory for all chassis. """

        for chassis in self.chassis_list.values():
            chassis.inventory(modules_inventory=True)