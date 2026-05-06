def add_primary_key(self, column="id"):
        """Add primary key constraint to specified column
        """
        if not self.primary_key:
            sql = """ALTER TABLE {s}.{t}
                     ADD PRIMARY KEY ({c})
                  """.format(
                s=self.schema, t=self.name, c=column
            )
            self.db.execute(sql)