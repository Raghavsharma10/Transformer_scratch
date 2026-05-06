def _set_metadata(self):
        """
        Internal helper to set metadata attributes.
        """
        meta = QueryDbMeta()
        with self._engine.connect() as conn:
            meta.bind = conn
            meta.reflect()
            self._meta = meta

        # Set an inspect attribute, whose subattributes
        # return individual tables / columns. Tables and columns
        # are special classes with .last() and other convenience methods
        self.inspect = QueryDbAttributes()
        for table in self._meta.tables:
            setattr(self.inspect, table,
                    QueryDbOrm(self._meta.tables[table], self))

            table_attr = getattr(self.inspect, table)
            table_cols = table_attr.table.columns

            for col in table_cols.keys():
                setattr(table_attr, col,
                        QueryDbOrm(table_cols[col], self))

            # Finally add some summary info:
            #   Table name
            #   Primary Key item or list
            #   N of Cols
            #   Distinct Col Values (class so NVARCHAR(20) and NVARCHAR(30) are not different)
            primary_keys = table_attr.table.primary_key.columns.keys()
            self._summary_info.append((
                table,
                primary_keys[0] if len(primary_keys) == 1 else primary_keys,
                len(table_cols),
                len(set([x.type.__class__ for x in table_cols.values()])),
                ))