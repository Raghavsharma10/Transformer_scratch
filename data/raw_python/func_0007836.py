def aspectBenefics(self):
        """ Returns a list with the good aspects the object 
        makes to the benefics.
        
        """
        benefics = [const.VENUS, const.JUPITER]
        return self.__aspectLists(benefics, aspList=[0, 60, 120])