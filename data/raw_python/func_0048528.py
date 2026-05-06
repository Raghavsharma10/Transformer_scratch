def _update_object_map(self, obj_map):
        """unclear if it's better to use this method or get_object_map
        My main consideration is that MultiLanguageQuestionRecord already
        overrides get_object_map
        """
        obj_map['droppables'] = self.get_droppables()
        obj_map['targets'] = self.get_targets()
        obj_map['zones'] = self.get_zones()