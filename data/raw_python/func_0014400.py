def get_groups_count(self, field=None):
        '''
        Returns 'matches' from group response.

                If grouping on more than one field, provide the field argument to specify which count you are looking for.
        '''
        field = field if field else self._determine_group_field(field)
        if 'matches' in self.data['grouped'][field]:
            return self.data['grouped'][field]['matches']
        raise ValueError("group matches not found in response")