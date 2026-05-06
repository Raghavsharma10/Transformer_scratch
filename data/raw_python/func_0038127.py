def get_traffic_items(self):
        """
        :return: dictionary {name: object} of all traffic items.
        """

        traffic = self.get_child_static('traffic')
        return {o.obj_name(): o for o in traffic.get_objects_or_children_by_type('trafficItem')}