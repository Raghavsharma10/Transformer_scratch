def clear_target_names(self, target_id):
        """stub"""
        if self.get_targets_metadata().is_read_only():
            raise NoAccess()
        updated_targets = []
        for current_target in self.my_osid_object_form._my_map['targets']:
            if current_target['id'] != target_id:
                updated_targets.append(current_target)
            else:
                updated_targets.append({
                    'id': current_target['id'],
                    'texts': current_target['texts'],
                    'names': [],
                    'dropBehaviorType': current_target['dropBehaviorType']
                })
        self.my_osid_object_form._my_map['targets'] = updated_targets