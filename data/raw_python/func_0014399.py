def get_ngroups(self, field=None):
        '''
        Returns ngroups count if it was specified in the query, otherwise ValueError.

        If grouping on more than one field, provide the field argument to specify which count you are looking for.
        '''
        field = field if field else self._determine_group_field(field)
        if 'ngroups' in self.data['grouped'][field]:
            return self.data['grouped'][field]['ngroups']
        raise ValueError("ngroups not found in response. specify group.ngroups in the query.")