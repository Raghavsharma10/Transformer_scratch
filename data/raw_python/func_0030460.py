def inflate(self, amount, target=datetime.date.today(), reference=None,
                country=None):
        """
        Inflate a given amount to the target date from a reference date (or
        the object's reference year if no reference date is provided) in a
        given country (or objects country if no country is provided). The
        amount has to be provided as it was valued in the reference year.
        """
        country = country if country else self.country

        # Get the inflation for the two dates and country
        inflation = self.get(reference, country, target)
        # Return the inflated/deflated amount
        return amount * inflation.factor