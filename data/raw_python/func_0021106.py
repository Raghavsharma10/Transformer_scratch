def restore_descriptor(self, dataframe):
        """Restore descriptor from Pandas
        """

        # Prepare
        fields = []
        primary_key = None

        # Primary key
        if dataframe.index.name:
            field_type = self.restore_type(dataframe.index.dtype)
            field = {
                'name': dataframe.index.name,
                'type': field_type,
                'constraints': {'required': True},
            }
            fields.append(field)
            primary_key = dataframe.index.name

        # Fields
        for column, dtype in dataframe.dtypes.iteritems():
            sample = dataframe[column].iloc[0] if len(dataframe) else None
            field_type = self.restore_type(dtype, sample=sample)
            field = {'name': column, 'type': field_type}
            # TODO: provide better required indication
            # if dataframe[column].isnull().sum() == 0:
            #     field['constraints'] = {'required': True}
            fields.append(field)

        # Descriptor
        descriptor = {}
        descriptor['fields'] = fields
        if primary_key:
            descriptor['primaryKey'] = primary_key

        return descriptor