def do_use(self, line):
        "use {tablename}"
        self.table = self.conn.get_table(line)
        self.pprint(self.conn.describe_table(self.table.name))
        self.prompt = "%s> " % self.table.name