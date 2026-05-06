def from_dataframe(cls, name, df, indices, primary_key=None):
        """Infer table metadata from a DataFrame"""

        # ordered list (column_name, column_type) pairs
        column_types = []
        # which columns have nullable values
        nullable = set()

        # tag cached database by dataframe's number of rows and columns
        for column_name in df.columns:
            values = df[column_name]
            if values.isnull().any():
                nullable.add(column_name)
            column_db_type = db_type(values.dtype)
            column_types.append((column_name.replace(" ", "_"), column_db_type))

        def make_rows():
            return list(tuple(row) for row in df.values)

        return cls(
            name=name,
            column_types=column_types,
            make_rows=make_rows,
            indices=indices,
            nullable=nullable,
            primary_key=primary_key)