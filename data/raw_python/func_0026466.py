def cli_schema(self, *args):
        """Display a single schema definition"""

        key = None
        if len(args) > 1:
            key = args[1]
            args = list(args)

        if '-config' in args or '-c' in args:
            store = configschemastore
            try:
                args.remove('-c')
                args.remove('-config')
            except ValueError:
                pass
        else:
            store = schemastore

        def output(schema):
            self.log("%s :" % schema)
            if key == 'props':
                self.log(store[schema]['schema']['properties'], pretty=True)
            elif key == 'perms':
                try:
                    self.log(store[schema]['schema']['roles_create'], pretty=True)
                except KeyError:
                    self.log('Schema', schema, 'has no role for creation', lvl=warn)
                try:
                    self.log(store[schema]['schema']['properties']['perms']['properties'], pretty=True)
                except KeyError:
                    self.log('Schema', schema, 'has no permissions', lvl=warn)
            else:
                self.log(store[schema]['schema'], pretty=True)

        if '*' in args:
            for schema in store:
                output(schema)
        else:
            output(args[0])