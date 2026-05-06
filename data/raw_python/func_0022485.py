def from_dict(self, d):
        """
        Create a Stage from a dictionary. The change is in inplace.

        :argument: python dictionary
        :return: None
        """

        if 'uid' in d:
            if d['uid']:
                self._uid = d['uid']

        if 'name' in d:
            if d['name']:
                self._name = d['name']

        if 'state' in d:
            if isinstance(d['state'], str) or isinstance(d['state'], unicode):
                if d['state'] in states._stage_state_values.keys():
                    self._state = d['state']
                else:
                    raise ValueError(obj=self._uid,
                                     attribute='state',
                                     expected_value=states._stage_state_values.keys(),
                                     actual_value=value)
            else:
                raise TypeError(entity='state', expected_type=str, actual_type=type(d['state']))

        else:
            self._state = states.INITIAL

        if 'state_history' in d:
            if isinstance(d['state_history'], list):
                self._state_history = d['state_history']
            else:
                raise TypeError(entity='state_history', expected_type=list, actual_type=type(d['state_history']))

        if 'parent_pipeline' in d:
            if isinstance(d['parent_pipeline'], dict):
                self._p_pipeline = d['parent_pipeline']
            else:
                raise TypeError(entity='parent_pipeline', expected_type=dict, actual_type=type(d['parent_pipeline']))