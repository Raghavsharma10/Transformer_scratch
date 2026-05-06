def do_use(self, line):
        "use {tablename}"
        self.table = boto.dynamodb2.table.Table(line, connection=self.conn)
        self.pprint(self.table.describe())
        self.prompt = "%s> " % self.table.table_name