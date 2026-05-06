def _get_generic_schema(self):
        """ Returns whoosh's generic schema. """
        schema = Schema(
            identifier=ID(stored=True),  # Partition versioned id
            type=ID(stored=True),
            name=NGRAM(phrase=True, stored=True, minsize=2, maxsize=8))
        return schema