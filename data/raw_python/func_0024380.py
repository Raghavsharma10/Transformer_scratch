def copy(self):
        """
        get copy of object

        :return: ReactionContainer
        """
        return type(self)(reagents=[x.copy() for x in self.__reagents], meta=self.__meta.copy(),
                          products=[x.copy() for x in self.__products],
                          reactants=[x.copy() for x in self.__reactants])