def getDignities(self):
        """ Returns the dignities belonging to this object. """
        info = self.getInfo()
        dignities = [dign for (dign, objID) in info.items()
                        if objID == self.obj.id]
        return dignities