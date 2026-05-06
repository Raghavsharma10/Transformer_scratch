def add_shex(self, schema: str) -> "PrefixLibrary":
        """ Add a ShExC schema to the library

        :param schema: ShExC schema text, URL or file name
        :return: prefix library object
        """
        if '\n' in schema or '\r' in schema or ' ' in schema:
            shex = schema
        else:
            shex = load_shex_file(schema)

        for line in shex.split('\n'):
            line = line.strip()
            m = re.match(r'PREFIX\s+(\S+):\s+<(\S+)>', line)
            if not m:
                m = re.match(r"@prefix\s+(\S+):\s+<(\S+)>\s+\.", line)
            if m:
                setattr(self, m.group(1).upper(), Namespace(m.group(2)))
        return self