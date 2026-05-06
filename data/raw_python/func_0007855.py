def getObjectsInHouse(self, house):
        """ Returns a list with all objects in a house. """
        res = [obj for obj in self if house.hasObject(obj)]
        return ObjectList(res)