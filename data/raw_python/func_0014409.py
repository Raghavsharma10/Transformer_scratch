def json_facet(self, field=None):
        '''
        EXPERIMENTAL

        Tried to kick back the json.fact output.
        '''
        facets = self.data['facets']
        if field is None:
            temp_fields = [x for x in facets.keys() if x != 'count']
            if len(temp_fields) != 1:
                raise ValueError("field argument not specified and it looks like there is more than one field in facets. Specify the field to get json.facet from. ")
            field = temp_fields[0]

        if field not in self.data['facets']:
            raise ValueError("Facet Field {} Not found in response, available fields are {}".format(
                                        field, self.data['facets'].keys() ))
        return self.data['facets'][field]