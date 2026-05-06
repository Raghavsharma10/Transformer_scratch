def add_iterative_predicate(self, key, values_list):
        """add an iterative predicate with a key and set of values
        which it can be equal to in and or function.
        The individual predicates are specified with the type ``equals`` and
        combined with a type ``or``.

        The main reason for this addition is the inability of using ``in`` as
        predicate type wfor multiple taxon_key values
        (cfr. http://dev.gbif.org/issues/browse/POR-2753)

        :param key: API key to use for the query.
        :param values_list: Filename or list containing the taxon keys to be s
            searched.

        """
        values = self._extract_values(values_list)

        predicate = {'type': 'equals', 'key': key, 'value': None}
        predicates = []
        while values:
            predicate['value'] = values.pop()
            predicates.append(predicate.copy())
        self.predicates.append({'type': 'or', 'predicates': predicates})