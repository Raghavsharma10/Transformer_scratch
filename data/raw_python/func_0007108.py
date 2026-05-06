def select_by_critere(self, base, criteria):
        """
        :param base: Reference on whole base
        :param criteria: Callable abstractAcces -> Bool, acting as filter
        :return: Collection on acces passing the criteria
        """
        Ac = self.ACCES
        return groups.Collection(Ac(base, i) for i in self if criteria(Ac(base, i)))