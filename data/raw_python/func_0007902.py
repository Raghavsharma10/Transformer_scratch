def getUTC(self):
        """ Returns this Datetime localized for UTC. """
        timeUTC = self.time.getUTC(self.utcoffset)
        dateUTC = Date(round(self.jd))
        return Datetime(dateUTC, timeUTC)