def get(self, key):
        """
        simple `select`
        """
        if not Log:
            _late_import()
        return FlatList(vals=[unwrap(coalesce(_datawrap(v), Null)[key]) for v in _get_list(self)])