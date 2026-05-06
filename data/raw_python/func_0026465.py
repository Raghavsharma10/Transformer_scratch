def cli_form(self, *args):
        """Display a schemata's form definition"""

        if args[0] == '*':
            for schema in schemastore:
                self.log(schema, ':', schemastore[schema]['form'], pretty=True)
        else:
            self.log(schemastore[args[0]]['form'], pretty=True)