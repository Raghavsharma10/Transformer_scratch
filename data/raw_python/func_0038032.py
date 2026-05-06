def _schema_getter(self, db):
        """Retrieve a dictionary representing a database's data schema."""
        # Change DB connection if needed
        if self.database != db:
            self.change_db(db)
        schema_dict = {tbl: self.get_schema(tbl) for tbl in self.tables}

        schema_lst = []
        for table, schema in schema_dict.items():
            for col in schema:
                col.insert(0, table)
                schema_lst.append(col)
        return schema_lst