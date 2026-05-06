def rename(self, name):
        """Rename the table
        """
        sql = """ALTER TABLE {s}.{t} RENAME TO {name}
              """.format(
            s=self.schema, t=self.name, name=name
        )
        self.engine.execute(sql)
        self.table = SQLATable(name, self.metadata, schema=self.schema, autoload=True)