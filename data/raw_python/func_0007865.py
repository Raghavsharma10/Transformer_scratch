def N(self, ID, asp=0):
        """ Returns the conjunction or opposition aspect 
        of an object. 
        
        """
        obj = self.chart.get(ID).copy()
        obj.relocate(obj.lon + asp)
        ID = 'N_%s_%s' % (ID, asp)
        return self.G(ID, obj.lat, obj.lon)