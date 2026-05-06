def _get_generic_schema(self):
        """ Returns whoosh's generic schema of the dataset. """
        schema = Schema(
            vid=ID(stored=True, unique=True),  # Object id
            title=NGRAMWORDS(),
            keywords=KEYWORD,  # Lists of coverage identifiers, ISO time values and GVIDs, source names, source abbrev
            doc=TEXT)  # Generated document for the core of the topic search
        return schema