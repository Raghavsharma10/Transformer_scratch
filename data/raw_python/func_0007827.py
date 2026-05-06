def house(self):
        """ Returns the object's house. """
        house = self.chart.houses.getObjectHouse(self.obj)
        return house