def getFixedStars(self):
        """ Returns a list with all fixed stars. """
        IDs = const.LIST_FIXED_STARS
        return ephem.getFixedStarList(IDs, self.date)