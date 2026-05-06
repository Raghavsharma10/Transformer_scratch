def sunRelation(self):
        """ Returns the relation of the object with the sun. """
        sun = self.chart.getObject(const.SUN)
        return sunRelation(self.obj, sun)