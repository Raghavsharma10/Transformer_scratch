def does_field_exist(self,collection,field_name):
        '''
        Checks if the field exists will return a boolean True (exists) or False(doesn't exist).

        :param string collection: Name of the collection for the action
        :param string field_name: String name of the field.
        '''
        schema = self.get_schema_fields(collection)
        logging.info(schema)
        return True if field_name in [field['name'] for field in schema['fields']] else False