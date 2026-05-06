def D(self, ID, asp):
        """ Returns the dexter aspect of an object. """
        obj = self.chart.getObject(ID).copy()
        obj.relocate(obj.lon - asp)
        ID = 'D_%s_%s' % (ID, asp)
        return self.G(ID, obj.lat, obj.lon)