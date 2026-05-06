def C(self, ID):
        """ Returns the CAntiscia of an object. """
        obj = self.chart.getObject(ID).cantiscia()
        ID = 'C_%s' % (ID)
        return self.G(ID, obj.lat, obj.lon)