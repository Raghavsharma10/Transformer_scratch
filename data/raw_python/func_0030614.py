def get(self, date=datetime.date.today(), country=None):
        """
        Get the CPI value for a specific time. Defaults to today. This uses
        the closest method internally but sets limit to one day.
        """
        if not country:
            country = self.country
        if country == "all":
            raise ValueError("You need to specify a country")
        if not isinstance(date, str) and not isinstance(date, int):
            date = date.year

        cpi = self.data.get(country.upper(), {}).get(str(date))
        if not cpi:
            raise ValueError("Missing CPI data for {} for {}".format(
                country, date))

        return CPIResult(date=date, value=cpi)