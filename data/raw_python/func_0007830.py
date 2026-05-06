def orientality(self):
        """ Returns the orientality of the object. """
        sun = self.chart.getObject(const.SUN)
        return orientality(self.obj, sun)