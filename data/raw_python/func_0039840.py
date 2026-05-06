def _tables(self) -> List[Tuple[str, str]]:
        """
        Return a list of all known tables and and its full URI
        :return: table name and full URI
        """
        return [(k.rsplit('.', 1)[1] if '.' in k else k, k) for k in self._ont_tables.keys()]