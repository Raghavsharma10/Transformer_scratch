def new_db_from_pandas(self, frame, table=None, data=None, load=True, **kwargs):
        """Create a new db partition from a pandas data frame.

        If the table does not exist, it will be created

        """

        from ..orm import Column
        # from dbexceptions import ConfigurationError

        # Create the table from the information in the data frame.
        with self.bundle.session:
            sch = self.bundle.schema
            t = sch.new_table(table)

            if frame.index.name:
                id_name = frame.index.name
            else:
                id_name = 'id'

            sch.add_column(t, id_name,
                           datatype=Column.convert_numpy_type(frame.index.dtype),
                           is_primary_key=True)

            for name, type_ in zip([row for row in frame.columns],
                                   [row for row in frame.convert_objects(convert_numeric=True,
                                                                         convert_dates=True).dtypes]):
                sch.add_column(t, name, datatype=Column.convert_numpy_type(type_))
                sch.write_schema()

        p = self.new_partition(table=table, data=data, **kwargs)

        if load:
            pk_name = frame.index.name
            with p.inserter(table) as ins:
                for i, row in frame.iterrows():
                    d = dict(row)
                    d[pk_name] = i

                    ins.insert(d)
        return p