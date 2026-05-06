def do_create(self, line):
        "create {tablename} [-c rc,wc] {hkey}[:{type} {rkey}:{type}]"
        args = self.getargs(line)
        rc = wc = 5

        name = args.pop(0)  # tablename

        if args[0] == "-c":  # capacity
            args.pop(0)  # skyp -c

            capacity = args.pop(0).strip()
            rc, _, wc = capacity.partition(",")
            rc = int(rc)
            wc = int(wc) if wc != "" else rc

        hkey, _, hkey_type = args.pop(0).partition(':')
        hkey_type = self.get_type(hkey_type or 'S')

        if args:
            rkey, _, rkey_type = args.pop(0).partition(':')
            rkey_type = self.get_type(rkey_type or 'S')
        else:
            rkey = rkey_type = None

        t = self.conn.create_table(name,
                                   self.conn.create_schema(hkey, hkey_type, rkey, rkey_type),
                                   rc, wc)
        self.pprint(self.conn.describe_table(t.name))