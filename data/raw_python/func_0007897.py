def getUTC(self, utcoffset):
        """ Returns a new Time object set to UTC given 
        an offset Time object.
        
        """
        newTime = (self.value - utcoffset.value) % 24
        return Time(newTime)