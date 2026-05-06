def get_first_field_values_as_list(self, field):
        '''
        :param str field: The name of the field for lookup.

        Goes through all documents returned looking for specified field. At first encounter will return the field's value.
        '''
        for doc in self.docs:
            if field in doc.keys():
                return doc[field]
        raise SolrResponseError("No field in result set")