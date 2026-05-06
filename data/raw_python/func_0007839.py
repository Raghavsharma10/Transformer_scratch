def isAuxilied(self):
        """ Returns if the object is separating and applying to 
        a benefic considering good aspects.
        
        """
        benefics = [const.VENUS, const.JUPITER]
        return self.__sepApp(benefics, aspList=[0, 60, 120])