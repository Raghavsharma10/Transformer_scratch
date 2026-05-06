def all(self):
        """ Returns list with all indexed identifiers. """
        identifiers = []

        query = text("""
            SELECT identifier, type, name
            FROM identifier_index;""")

        for result in self.execute(query):
            vid, type_, name = result
            res = IdentifierSearchResult(
                score=1, vid=vid, type=type_, name=name)
            identifiers.append(res)
        return identifiers