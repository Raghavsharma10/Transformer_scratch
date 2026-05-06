def do_refresh(self, line):
        "refresh {table_name}"
        table = self.get_table(line)
        table.refresh(True)
        self.pprint(self.conn.describe_table(table.name))