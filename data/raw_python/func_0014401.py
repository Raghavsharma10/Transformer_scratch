def get_flat_groups(self, field=None):
        '''
        Flattens the group response and just returns a list of documents.
        '''
        field = field if field else self._determine_group_field(field)
        temp_groups = self.data['grouped'][field]['groups']
        return [y for x in temp_groups for y in x['doclist']['docs']]