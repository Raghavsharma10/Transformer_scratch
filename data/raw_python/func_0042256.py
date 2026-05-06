def do_drop(self, line):
        "drop {tablename}"
        self.conn.delete_table(self.conn.get_table(line))