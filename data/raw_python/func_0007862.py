def A(self, ID):
        """ Returns the Antiscia of an object. """
        obj = self.chart.getObject(ID).antiscia()
        ID = 'A_%s' % (ID)
        return self.G(ID, obj.lat, obj.lon)