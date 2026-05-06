def create_datastore_from_json_schema(self, json_path, delete_first=0, path=None):
        # type: (str, int, Optional[str]) -> None
        """For tabular data, create a resource in the HDX datastore which enables data preview in HDX from a JSON file
        containing a list of fields and types of form {'id': 'FIELD', 'type': 'TYPE'} and optionally a primary key.
        If path is not supplied, the file is first downloaded from HDX.

        Args:
            json_path (str): Path to JSON file containing list of fields and types of form {'id': 'FIELD', 'type': 'TYPE'}
            delete_first (int): Delete datastore before creation. 0 = No, 1 = Yes, 2 = If no primary key. Defaults to 0.
            path (Optional[str]): Local path to file that was uploaded. Defaults to None.

        Returns:
            None
        """
        data = load_json(json_path)
        self.create_datastore_from_dict_schema(data, delete_first, path=path)