def convert_descriptor_and_rows(self, descriptor, rows):
        """Convert descriptor and rows to Pandas
        """

        # Prepare
        primary_key = None
        schema = tableschema.Schema(descriptor)
        if len(schema.primary_key) == 1:
            primary_key = schema.primary_key[0]
        elif len(schema.primary_key) > 1:
            message = 'Multi-column primary keys are not supported'
            raise tableschema.exceptions.StorageError(message)

        # Get data/index
        data_rows = []
        index_rows = []
        jtstypes_map = {}
        for row in rows:
            values = []
            index = None
            for field, value in zip(schema.fields, row):
                try:
                    if isinstance(value, float) and np.isnan(value):
                        value = None
                    if value and field.type == 'integer':
                        value = int(value)
                    value = field.cast_value(value)
                except tableschema.exceptions.CastError:
                    value = json.loads(value)
                # http://pandas.pydata.org/pandas-docs/stable/gotchas.html#support-for-integer-na
                if value is None and field.type in ('number', 'integer'):
                    jtstypes_map[field.name] = 'number'
                    value = np.NaN
                if field.name == primary_key:
                    index = value
                else:
                    values.append(value)
            data_rows.append(tuple(values))
            index_rows.append(index)

        # Get dtypes
        dtypes = []
        for field in schema.fields:
            if field.name != primary_key:
                field_name = field.name
                if six.PY2:
                    field_name = field.name.encode('utf-8')
                dtype = self.convert_type(jtstypes_map.get(field.name, field.type))
                dtypes.append((field_name, dtype))

        # Create dataframe
        index = None
        columns = schema.headers
        array = np.array(data_rows, dtype=dtypes)
        if primary_key:
            index_field = schema.get_field(primary_key)
            index_dtype = self.convert_type(index_field.type)
            index_class = pd.Index
            if index_field.type in ['datetime', 'date']:
                index_class = pd.DatetimeIndex
            index = index_class(index_rows, name=primary_key, dtype=index_dtype)
            columns = filter(lambda column: column != primary_key, schema.headers)
        dataframe = pd.DataFrame(array, index=index, columns=columns)

        return dataframe