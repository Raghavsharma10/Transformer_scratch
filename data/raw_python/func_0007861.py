def T(self, ID, sign):
        """ Returns the term of an object in a sign. """
        lon = self.terms[sign][ID]
        ID = 'T_%s_%s' % (ID, sign)
        return self.G(ID, 0, lon)