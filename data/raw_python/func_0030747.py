def create_table(table, connection, schema=None):
        """Create a single table, primarily used din migrations"""

        orig_schemas = {}

        # These schema shenanigans are almost certainly wrong.
        # But they are expedient. For Postgres, it puts the library
        # tables in the Library schema. We need to change the schema for all tables in case
        # the table we are creating references another table
        if schema:
            connection.execute("SET search_path TO {}".format(schema))

            for table in ALL_TABLES:
                orig_schemas[table.__table__] = table.__table__.schema
                table.__table__.schema = schema

        table.__table__.create(bind=connection.engine)

        # We have to put the schemas back because when installing to a warehouse.
        # the same library classes can be used to access a Sqlite database, which
        # does not handle schemas.
        if schema:
            for it, orig_schema in list(orig_schemas.items()):
                it.schema = orig_schema