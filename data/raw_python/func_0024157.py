def __get_mapping(self, structures):
        """
        match each pattern to each molecule.
        if all patterns matches with all molecules
        return generator of all possible mapping.

        :param structures: disjoint molecules
        :return: mapping generator
        """
        for c in permutations(structures, len(self.__patterns)):
            for m in product(*(x.get_substructure_mapping(y, limit=0) for x, y in zip(self.__patterns, c))):
                mapping = {}
                for i in m:
                    mapping.update(i)
                if mapping:
                    yield mapping