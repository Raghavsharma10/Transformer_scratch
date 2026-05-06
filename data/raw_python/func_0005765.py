def to_sql(self, url, table_name, *, schema=None, fields=None, fields_to_explode=None, if_exists="fail", dtype=None):
        """
        Export items as rows in a PostgreSQL table.

        Parameters
        ----------

        url: string
            Connection string to connect to the database.
            Example: "postgresql://postgres@127.0.0.1:5432/testdb"

        table_name: string
            Name of the database table. Note that if this name contains a dot ('.')
            and `schema` is not specified, the first part of the name before the dot
            will be interpreted as the schema name.

        schema : string, optional
            Specify the schema (if database flavor supports this). If None,
            use default schema or derive the schema name from `table_name`.

        fields: list or dict
            List of field names to export, or dictionary mapping output column names
            to attribute names of the generators.

            Examples:
               fields=['field_name_1', 'field_name_2']
               fields={'COL1': 'field_name_1', 'COL2': 'field_name_2'}

        fields_to_explode: list or None
            Optional list of field names where each entry (which must itself be a sequence)
            is to be "exploded" into separate rows.

        if_exists : {'fail', 'do_nothing', 'replace', 'append'}, default 'fail'
            - fail: If table exists, raise an error.
            - do_nothing: If table exists, do nothing and immediately return.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.

        dtype : dict, optional
            Specifying the datatype for columns. The keys should be the column
            names and the values should be the SQLAlchemy types or strings for
            the sqlite3 legacy mode. This is passed through to pandas.DataFrame.to_sql().
        """
        if schema is None:
            schema, table_name = _extract_schema_if_given(table_name)

        engine = create_engine(url)
        ins = inspect(engine)

        if schema is not None and schema not in ins.get_schema_names():
            logger.debug(f"Creating non-existing schema: '{schema}'")
            engine.execute(CreateSchema(schema))

        if table_name in ins.get_table_names(schema=schema) and if_exists == 'do_nothing':
            logger.debug("Table already exists (use if_exists='replace' or if_exists='append' to modify it).")
            return

        if if_exists == 'do_nothing':
            # we handled the 'do nothing' case above; change to an option that pandas will understand
            if_exists = 'fail'

        with engine.begin() as conn:
            self.to_df(fields=fields, fields_to_explode=fields_to_explode).to_sql(
                table_name, conn, schema=schema, index=False, if_exists=if_exists, dtype=dtype)