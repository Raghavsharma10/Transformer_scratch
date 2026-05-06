def isVOC(self, ID):
        """ Returns if a planet is Void of Course.
        A planet is not VOC if has any exact or applicative aspects
        ignoring the sign status (associate or dissociate).
        
        """
        asps = self.aspectsByCat(ID, const.MAJOR_ASPECTS)
        applications = asps[const.APPLICATIVE]
        exacts = asps[const.EXACT]
        return len(applications) == 0 and len(exacts) == 0