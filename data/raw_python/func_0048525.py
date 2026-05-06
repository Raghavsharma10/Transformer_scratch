def get_droppables(self):
        """stub"""
        droppables = []
        for current_droppable in self.my_osid_object._my_map['droppables']:
            droppables.append({
                'id': current_droppable['id'],
                'text': self.get_matching_language_value('texts',
                                                         dictionary=current_droppable).text,
                'name': self.get_matching_language_value('names',
                                                         dictionary=current_droppable).text,
                'reuse': current_droppable['reuse'],
                'dropBehaviorType': current_droppable['dropBehaviorType']
            })
        return droppables