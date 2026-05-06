def clear_droppable_texts(self, droppable_id):
        """stub"""
        if self.get_droppables_metadata().is_read_only():
            raise NoAccess()
        updated_droppables = []
        for current_droppable in self.my_osid_object_form._my_map['droppables']:
            if current_droppable['id'] != droppable_id:
                updated_droppables.append(current_droppable)
            else:
                updated_droppables.append({
                    'id': current_droppable['id'],
                    'texts': [],
                    'names': current_droppable['names'],
                    'reuse': current_droppable['reuse'],
                    'dropBehaviorType': current_droppable['dropBehaviorType']
                })
        self.my_osid_object_form._my_map['droppables'] = updated_droppables