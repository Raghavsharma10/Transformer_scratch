def set_target_order(self, target_ids):
        """ reorder targets per the passed in list
        :param target_ids:
        :return:
        """
        reordered_targets = []
        current_target_ids = [t['id'] for t in self.my_osid_object_form._my_map['targets']]
        if set(target_ids) != set(current_target_ids):
            raise IllegalState('target_ids do not match existing targets')

        for target_id in target_ids:
            for current_target in self.my_osid_object_form._my_map['targets']:
                if target_id == current_target['id']:
                    reordered_targets.append(current_target)
                    break

        self.my_osid_object_form._my_map['targets'] = reordered_targets