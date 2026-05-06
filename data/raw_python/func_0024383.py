def compose(self):
        """
        get CGR of reaction

        reagents will be presented as unchanged molecules
        :return: CGRContainer
        """
        rr = self.__reagents + self.__reactants
        if rr:
            if not all(isinstance(x, (MoleculeContainer, CGRContainer)) for x in rr):
                raise TypeError('Queries not composable')
            r = reduce(or_, rr)
        else:
            r = MoleculeContainer()
        if self.__products:
            if not all(isinstance(x, (MoleculeContainer, CGRContainer)) for x in self.__products):
                raise TypeError('Queries not composable')
            p = reduce(or_, self.__products)
        else:
            p = MoleculeContainer()
        return r ^ p