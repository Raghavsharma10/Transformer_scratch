def get_targets(self):
        """stub"""
        targets = []
        for current_target in self.my_osid_object._my_map['targets']:
            targets.append({
                'id': current_target['id'],
                'text': self.get_matching_language_value('texts',
                                                         dictionary=current_target).text,
                'name': self.get_matching_language_value('names',
                                                         dictionary=current_target).text,
                'dropBehaviorType': current_target['dropBehaviorType']
            })
        return targets