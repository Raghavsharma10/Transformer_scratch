def update_datastore(self, schema=None, primary_key=None,
                         path=None):
        # type: (Optional[List[Dict]], Optional[str], Optional[str]) -> None
        """For tabular data, update a resource in the HDX datastore which enables data preview in HDX. If no schema is provided
        all fields are assumed to be text. If path is not supplied, the file is first downloaded from HDX.

        Args:
            schema (List[Dict]): List of fields and types of form {'id': 'FIELD', 'type': 'TYPE'}. Defaults to None.
            primary_key (Optional[str]): Primary key of schema. Defaults to None.
            path (Optional[str]): Local path to file that was uploaded. Defaults to None.

        Returns:
            None
        """
        self.create_datastore(schema, primary_key, 2, path=path)