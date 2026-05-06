def ingest(self, **kwargs):

        '''
            a core method to ingest and validate arbitrary keyword data

            **NOTE: data is always returned with this method**

            for each key in the model, a value is returned according
             to the following priority:

                1. value in kwargs if field passes validation test
                2. default value declared for the key in the model
                3. empty value appropriate to datatype of key in the model

            **NOTE: as long as a default value is provided for each key-
             value, returned data will be model valid

            **NOTE: if 'extra_fields' is True for a dictionary, the key-
             value pair of all fields in kwargs which are not declared in
             the model will also be added to the corresponding dictionary
             data

            **NOTE: if 'max_size' is declared for a list, method will
             stop adding input to the list once it reaches max size

        :param kwargs: key, value pairs
        :return: dictionary with keys and value
        '''

        __name__ = '%s.ingest' % self.__class__.__name__

        schema_dict = self.schema
        path_to_root = '.'

        valid_data = self._ingest_dict(kwargs, schema_dict, path_to_root)

        return valid_data