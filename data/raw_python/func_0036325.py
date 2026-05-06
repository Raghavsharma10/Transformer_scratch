def breeding_female_location_type(self):
        """This attribute defines whether a female's current location is the same as the breeding cage to which it belongs.

        This attribute is used to color breeding table entries such that male mice which are currently in a different cage can quickly be identified.
        The location is relative to the first breeding cage an animal is assigned to."""
        try:
            self.breeding_females.all()[0].Cage
            if int(self.breeding_females.all()[0].Cage) == int(self.Cage):
                type = "resident-breeder"
            else:
                type = "non-resident-breeder"                
        except IndexError:
            type = "unknown-breeder"
        except ValueError:
            type = "unknown-breeder"            
        return type