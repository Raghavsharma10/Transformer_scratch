def _get_transformers(self):
        """Load the contents of meta_file and extract information about the transformers.

        Returns:
            dict: tuple(str, str) -> Transformer.
        """
        transformer_dict = {}

        for table in self.metadata['tables']:
            table_name = table['name']

            for field in table['fields']:
                transformer_type = field.get('type')
                if transformer_type:
                    col_name = field['name']
                    transformer_dict[(table_name, col_name)] = transformer_type

        return transformer_dict