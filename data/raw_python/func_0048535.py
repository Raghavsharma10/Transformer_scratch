def set_droppable_order(self, droppable_ids):
        """ reorder droppables per the passed in list
        :param droppable_ids:
        :return:
        """
        reordered_droppables = []
        current_droppable_ids = [d['id'] for d in self.my_osid_object_form._my_map['droppables']]
        if set(droppable_ids) != set(current_droppable_ids):
            raise IllegalState('droppable_ids do not match existing droppables')

        for droppable_id in droppable_ids:
            for current_droppable in self.my_osid_object_form._my_map['droppables']:
                if droppable_id == current_droppable['id']:
                    reordered_droppables.append(current_droppable)
                    break

        self.my_osid_object_form._my_map['droppables'] = reordered_droppables