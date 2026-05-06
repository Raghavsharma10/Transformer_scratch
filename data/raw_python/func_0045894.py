def set_values(self, choice_ids):
        """assume choice_ids is a list of choiceIds, like
        ["57978959cdfc5c42eefb36d1", "57978959cdfc5c42eefb36d0",
        "57978959cdfc5c42eefb36cf", "57978959cdfc5c42eefb36ce"]
        """
        # if not self.my_osid_object._my_map['choices']:
        #     raise IllegalState()
        organized_choices = []
        for choice_id in choice_ids:
            choice_obj = [c for c in self._original_choice_order if c['id'] == choice_id][0]
            organized_choices.append(choice_obj)
        self.my_osid_object._my_map['choices'] = organized_choices