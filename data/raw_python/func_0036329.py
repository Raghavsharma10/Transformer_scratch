def male_breeding_location_type(self):
        """This attribute defines whether a breeding male's current location is the same as the breeding cage.

        This attribute is used to color breeding table entries such that male mice which are currently in a different cage can quickly be identified."""
        if int(self.Male.all()[0].Cage) == int(self.Cage):
            type = "resident breeder"
        else:
            type = "non-resident breeder"
        return type