def inHouseJoy(self):
        """ Returns if the object is in its house of joy. """
        house = self.house()
        return props.object.houseJoy[self.obj.id] == house.id