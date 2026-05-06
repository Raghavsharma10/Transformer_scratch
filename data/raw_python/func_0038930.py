def check_data_type(self):
        """Check the type of the transformer and column match.

        Args:
            column_metadata(dict): Metadata of the column.

        Raises a ValueError if the types don't match
        """
        metadata_type = self.column_metadata.get('type')
        if self.type != metadata_type and metadata_type not in self.type:
            raise ValueError('Types of transformer don\'t match')