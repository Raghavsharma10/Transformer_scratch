def do_create(self, line):
        "create {tablename} [-c rc,wc] {hkey}[:{type} {rkey}:{type}]"
        args = self.getargs(line)
        rc = wc = 5

        name = args.pop(0)  #  tablename

        if args[0] == "-c": # capacity
            args.pop(0)  # skyp -c

            capacity = args.pop(0).strip()
            rc, _, wc = capacity.partition(",")
            rc = int(rc)
            wc = int(wc) if wc != "" else rc

        schema = []

        hkey, _, hkey_type = args.pop(0).partition(':')
        hkey_type = self.get_type(hkey_type or 'S')
        schema.append(boto.dynamodb2.fields.HashKey(hkey, hkey_type))

        if args:
            rkey, _, rkey_type = args.pop(0).partition(':')
            rkey_type = self.get_type(rkey_type or 'S')
            schema.append(boto.dynamodb2.fields.RangeKey(rkey, rkey_type))

        t = boto.dynamodb2.table.Table.create(name,
                                              schema=schema,
                                              throughput={'read': rc, 'write': wc})
        self.pprint(t.describe())