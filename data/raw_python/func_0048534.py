def remove_droppable(self, droppable_id):
        """remove a droppable, given the id"""
        updated_droppables = []
        for droppable in self.my_osid_object_form._my_map['droppables']:
            if droppable['id'] != droppable_id:
                updated_droppables.append(droppable)
        self.my_osid_object_form._my_map['droppables'] = updated_droppables