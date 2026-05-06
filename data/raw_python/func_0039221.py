def _get_tables(self, base_dir):
        """Load the contents of meta_file and the corresponding data.

        If fields containing Personally Identifiable Information are detected in the metadata
        they are anonymized before asign them into `table_dict`.

        Args:
            base_dir(str): Root folder of the dataset files.

        Returns:
            dict: Mapping str -> tuple(pandas.DataFrame, dict)
        """
        table_dict = {}

        for table in self.metadata['tables']:
            if table['use']:
                relative_path = os.path.join(base_dir, self.metadata['path'], table['path'])
                data_table = pd.read_csv(relative_path)
                pii_fields = self._get_pii_fields(table)
                data_table = self._anonymize_table(data_table, pii_fields)

                table_dict[table['name']] = (data_table, table)

        return table_dict