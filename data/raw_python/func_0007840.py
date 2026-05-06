def isSurrounded(self):
        """ Returns if the object is separating and applying to 
        a malefic considering bad aspects.
        
        """
        malefics = [const.MARS, const.SATURN]
        return self.__sepApp(malefics, aspList=[0, 90, 180])