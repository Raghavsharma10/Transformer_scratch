def remove_target(self, target_id):
        """remove a target, given the id"""
        updated_targets = []
        for target in self.my_osid_object_form._my_map['targets']:
            if target['id'] != target_id:
                updated_targets.append(target)
        self.my_osid_object_form._my_map['targets'] = updated_targets