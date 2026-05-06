def create_datastore_from_dict_schema(self, data, delete_first=0, path=None):
        # type: (dict, int, Optional[str]) -> None
        """For tabular data, create a resource in the HDX datastore which enables data preview in HDX from a dictionary
        containing a list of fields and types of form {'id': 'FIELD', 'type': 'TYPE'} and optionally a primary key.
        If path is not supplied, the file is first downloaded from HDX.

        Args:
            data (dict): Dictionary containing list of fields and types of form {'id': 'FIELD', 'type': 'TYPE'}
            delete_first (int): Delete datastore before creation. 0 = No, 1 = Yes, 2 = If no primary key. Defaults to 0.
            path (Optional[str]): Local path to file that was uploaded. Defaults to None.

        Returns:
            None
        """
        schema = data['schema']
        primary_key = data.get('primary_key')
        self.create_datastore(schema, primary_key, delete_first, path=path)