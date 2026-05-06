def update(self, command):
        """
        EXPECTING command == {"set":term, "clear":term, "where":where}
        THE set CLAUSE IS A DICT MAPPING NAMES TO VALUES
        THE where CLAUSE IS A JSON EXPRESSION FILTER
        """
        command = wrap(command)
        command_clear = listwrap(command["clear"])
        command_set = command.set.items()
        command_where = jx.get(command.where)

        for c in self.data:
            if command_where(c):
                for k in command_clear:
                    c[k] = None
                for k, v in command_set:
                    c[k] = v