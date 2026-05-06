def aspectMalefics(self):
        """ Returns a list with the bad aspects the object
        makes to the malefics.
        
        """
        malefics = [const.MARS, const.SATURN]
        return self.__aspectLists(malefics, aspList=[0, 90, 180])