def light(self):
        """ Returns if object is augmenting or diminishing its 
        light.
        
        """
        sun = self.chart.getObject(const.SUN)
        return light(self.obj, sun)