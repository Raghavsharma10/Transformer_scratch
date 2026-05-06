def create_datastore_for_topline(self, delete_first=0, path=None):
        # type: (int, Optional[str]) -> None
        """For tabular data, create a resource in the HDX datastore which enables data preview in HDX using the built in
        YAML definition for a topline. If path is not supplied, the file is first downloaded from HDX.

        Args:
            delete_first (int): Delete datastore before creation. 0 = No, 1 = Yes, 2 = If no primary key. Defaults to 0.
            path (Optional[str]): Local path to file that was uploaded. Defaults to None.

        Returns:
            None
        """
        data = load_yaml(script_dir_plus_file(join('..', 'hdx_datasource_topline.yml'), Resource))
        self.create_datastore_from_dict_schema(data, delete_first, path=path)