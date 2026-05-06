def get_field_values_as_list(self,field):
        '''
        :param str field: The name of the field for which to pull in values.
        Will parse the query results (must be ungrouped) and return all values of 'field' as a list. Note that these are not unique values.  Example::

            >>> r.get_field_values_as_list('product_name_exact')
            ['Mauris risus risus lacus. sit', 'dolor auctor Vivamus fringilla. vulputate', 'semper nisi lacus nulla sed', 'vel amet diam sed posuere', 'vitae neque ultricies, Phasellus ac', 'consectetur nisi orci, eu diam', 'sapien, nisi accumsan accumsan In', 'ligula. odio ipsum sit vel', 'tempus orci. elit, Ut nisl.', 'neque nisi Integer nisi Lorem']

        '''
        return [doc[field] for doc in self.docs if field in doc]